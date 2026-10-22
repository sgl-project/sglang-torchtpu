"""Tests for the TPU block-128/256/512 NVFP4 MoE weight contract.

The assertions are pure torch, but importing `fp4_moe` pulls in the Pallas MoE
package, which imports JAX eagerly. Register both lanes and skip on hosts
without JAX, matching `test_pallas_export.py`.
"""

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

BLOCK_SIZE = 512
SUPPORTED_BLOCK_SIZES = (16, 32, 64, 128, 256, BLOCK_SIZE)
EXPERTS = 2
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 512


def _make_block512_layer(
    include_unused_params: bool = True, group_size: int = BLOCK_SIZE
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True, activation="silu")
    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.randint(
                0,
                256,
                (EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE // 2),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.randint(
                0,
                256,
                (EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE // 2),
                dtype=torch.uint8,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.rand(
                EXPERTS,
                2 * INTERMEDIATE_SIZE,
                HIDDEN_SIZE // group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.rand(
                EXPERTS,
                HIDDEN_SIZE,
                INTERMEDIATE_SIZE // group_size,
                dtype=torch.float32,
            ),
            requires_grad=False,
        ),
    )
    if include_unused_params:
        for name in (
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            layer.register_parameter(
                name,
                torch.nn.Parameter(torch.ones(EXPERTS), requires_grad=False),
            )
    return layer


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
class TestNvfp4Block512Contract(CustomTestCase):
    def test_block512_layout_and_cleanup(self):
        from sglang_tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        layer = _make_block512_layer()
        expected_w13 = layer.w13_weight.data.permute(0, 2, 1).contiguous()
        expected_w2 = layer.w2_weight.data.permute(0, 2, 1).contiguous()
        expected_w13_scale = (
            layer.w13_weight_scale.data.permute(0, 2, 1).contiguous().unsqueeze(2)
        )
        expected_w2_scale = (
            layer.w2_weight_scale.data.permute(0, 2, 1).contiguous().unsqueeze(2)
        )

        prepare_tpu_nvfp4_moe_weights(layer)

        torch.testing.assert_close(layer.w13_weight, expected_w13)
        torch.testing.assert_close(layer.w2_weight, expected_w2)
        torch.testing.assert_close(layer.w13_weight_scale, expected_w13_scale)
        torch.testing.assert_close(layer.w2_weight_scale, expected_w2_scale)
        for name in (
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            self.assertIsNone(layer._parameters[name])

    def test_scale_layout_owns_its_declared_rank4_buffer(self):
        """The layout helper must return an independently backed rank-4 tensor.

        Appending a size-1 dimension to a contiguous rank-3 tensor produces a
        contiguous view. Shape, value, and contiguity checks therefore do not
        detect the alias; ownership must be asserted directly on the helper's
        return value.
        """
        from sglang_tpu.fp4_moe import (
            _to_gmm_v2_scale_layout,
        )

        for out_size in (2 * INTERMEDIATE_SIZE, HIDDEN_SIZE):
            for scale_dtype in (torch.float32, torch.bfloat16):
                with self.subTest(out_size=out_size, scale_dtype=scale_dtype):
                    source = torch.rand(EXPERTS, out_size, HIDDEN_SIZE // BLOCK_SIZE)

                    out = _to_gmm_v2_scale_layout(source, scale_dtype)

                    self.assertIsNone(out._base, "scale is still a view of its source")
                    self.assertTrue(out.is_contiguous())
                    self.assertEqual(
                        tuple(out.shape),
                        (EXPERTS, HIDDEN_SIZE // BLOCK_SIZE, 1, out_size),
                    )
                    expected = (
                        source.permute(0, 2, 1)
                        .contiguous()
                        .unsqueeze(2)
                        .to(scale_dtype)
                    )
                    self.assertTrue(torch.equal(out, expected))

    def test_rejects_unsupported_blocks_even_with_fused_rs(self):
        from sglang_tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        for group_size in (8, 96, 1024):
            for use_fused_rs in (False, True):
                with (
                    self.subTest(group_size=group_size, use_fused_rs=use_fused_rs),
                    self.assertRaisesRegex(NotImplementedError, "group_size in"),
                ):
                    prepare_tpu_nvfp4_moe_weights(
                        _make_block512_layer(),
                        group_size=group_size,
                        use_fused_rs=use_fused_rs,
                    )

    def test_sub512_rejects_stock_pallas_layout(self):
        from sglang_tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        for group_size in (16, 32, 64, 128, 256):
            with (
                self.subTest(group_size=group_size),
                self.assertRaisesRegex(NotImplementedError, "requires the fused-RS"),
            ):
                prepare_tpu_nvfp4_moe_weights(
                    _make_block512_layer(group_size=group_size),
                    group_size=group_size,
                )

    def test_scale_storage_dtype(self):
        from sglang_tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        layer = _make_block512_layer()
        prepare_tpu_nvfp4_moe_weights(layer, scale_dtype=torch.bfloat16)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.bfloat16)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.bfloat16)

    def test_folds_non_unit_global_scales(self):
        from sglang_tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        layer = _make_block512_layer()
        w13_block_scale = layer.w13_weight_scale.detach().clone()
        w2_block_scale = layer.w2_weight_scale.detach().clone()
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            torch.tensor([[2.0, 3.0], [4.0, 5.0]]), requires_grad=False
        )
        layer.w2_weight_scale_2 = torch.nn.Parameter(
            torch.tensor([6.0, 7.0]), requires_grad=False
        )
        expected_w13 = w13_block_scale
        expected_w13[:, :INTERMEDIATE_SIZE] *= torch.tensor([2.0, 4.0])[:, None, None]
        expected_w13[:, INTERMEDIATE_SIZE:] *= torch.tensor([3.0, 5.0])[:, None, None]
        expected_w13 = expected_w13.permute(0, 2, 1).contiguous().unsqueeze(2)
        expected_w2 = w2_block_scale * torch.tensor([6.0, 7.0])[:, None, None]
        expected_w2 = expected_w2.permute(0, 2, 1).contiguous().unsqueeze(2)

        prepare_tpu_nvfp4_moe_weights(layer)

        torch.testing.assert_close(layer.w13_weight_scale, expected_w13)
        torch.testing.assert_close(layer.w2_weight_scale, expected_w2)

    def test_fused_rs_requires_ep_only_reduced_combined_output(self):
        from sglang_tpu.fp4_moe import (
            can_use_tpu_nvfp4_fused_rs,
        )

        layer = types.SimpleNamespace(
            moe_ep_size=4,
            allow_backend_reduction=True,
            num_fused_shared_experts=0,
            moe_runner_config=types.SimpleNamespace(
                is_gated=True,
                activation="silu",
                apply_router_weight_on_input=False,
            ),
        )
        tp_group = types.SimpleNamespace(world_size=4)
        world_group = types.SimpleNamespace(world_size=4)
        parallel = types.SimpleNamespace(
            attn_dp_size=1,
            enable_dp_attention=False,
            attn_dp_enabled=False,
            tp_group=tp_group,
            world_group=world_group,
        )
        exec_ns = types.SimpleNamespace(
            moe=types.SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
        )
        a2a_backend = types.SimpleNamespace(is_none=lambda: True)
        with (
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ),
            mock.patch("sglang.srt.runtime_context.get_exec", return_value=exec_ns),
            mock.patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=a2a_backend,
            ),
        ):
            for group_size in SUPPORTED_BLOCK_SIZES:
                with self.subTest(group_size=group_size):
                    self.assertTrue(
                        can_use_tpu_nvfp4_fused_rs(
                            layer, group_size=group_size, no_combine=False
                        )
                    )
            layer.moe_ep_size = 2
            self.assertFalse(
                can_use_tpu_nvfp4_fused_rs(
                    layer, group_size=BLOCK_SIZE, no_combine=False
                )
            )
            layer.moe_ep_size = 4
            parallel.attn_dp_size = 2
            self.assertFalse(
                can_use_tpu_nvfp4_fused_rs(
                    layer, group_size=BLOCK_SIZE, no_combine=False
                )
            )
            parallel.attn_dp_size = 1
            layer.allow_backend_reduction = False
            self.assertFalse(
                can_use_tpu_nvfp4_fused_rs(
                    layer, group_size=BLOCK_SIZE, no_combine=False
                )
            )
            layer.allow_backend_reduction = True
            # Without the shared post-expert norm contract, no_combine output
            # has no consumer, so fused-RS must decline it.
            self.assertFalse(
                can_use_tpu_nvfp4_fused_rs(
                    layer, group_size=BLOCK_SIZE, no_combine=True
                )
            )
            # With the contract declared the polarity flips: per-expert output
            # is required, and the combined form is the one that is rejected.
            layer.supports_tpu_fused_rs_post_expert_norm = True
            self.assertTrue(
                can_use_tpu_nvfp4_fused_rs(
                    layer, group_size=BLOCK_SIZE, no_combine=True
                )
            )
            self.assertFalse(
                can_use_tpu_nvfp4_fused_rs(
                    layer, group_size=BLOCK_SIZE, no_combine=False
                )
            )
            del layer.supports_tpu_fused_rs_post_expert_norm
            parallel.enable_dp_attention = True
            parallel.attn_dp_enabled = True
            self.assertFalse(
                can_use_tpu_nvfp4_fused_rs(
                    layer, group_size=BLOCK_SIZE, no_combine=False
                )
            )

    def test_unmet_requirements_name_the_failing_contract(self):
        """A rejected opt-in must say why, not silently serve the stock path.

        The no_combine mismatch below is the shape a model hits when it asks for
        per-expert output without declaring the post-expert-norm contract.
        """
        from sglang_tpu.fp4_moe import (
            TPU_NVFP4_SUPPORTED_GROUP_SIZES,
            unmet_tpu_nvfp4_fused_rs_requirements,
        )

        layer = types.SimpleNamespace(
            moe_ep_size=4,
            allow_backend_reduction=True,
            num_fused_shared_experts=0,
            moe_runner_config=types.SimpleNamespace(
                is_gated=True,
                activation="silu",
                apply_router_weight_on_input=False,
            ),
        )
        tp_group = types.SimpleNamespace(world_size=4)
        world_group = types.SimpleNamespace(world_size=4)
        parallel = types.SimpleNamespace(
            attn_dp_size=1,
            enable_dp_attention=False,
            attn_dp_enabled=False,
            tp_group=tp_group,
            world_group=world_group,
        )
        exec_ns = types.SimpleNamespace(
            moe=types.SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
        )
        a2a_backend = types.SimpleNamespace(is_none=lambda: True)
        with (
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ),
            mock.patch("sglang.srt.runtime_context.get_exec", return_value=exec_ns),
            mock.patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=a2a_backend,
            ),
        ):
            self.assertEqual(
                unmet_tpu_nvfp4_fused_rs_requirements(
                    layer, group_size=BLOCK_SIZE, no_combine=False
                ),
                [],
            )
            self.assertEqual(
                unmet_tpu_nvfp4_fused_rs_requirements(
                    layer, group_size=BLOCK_SIZE, no_combine=True
                ),
                ["no_combine matches the model's post-expert-norm contract"],
            )
            # Every unmet requirement is reported, not just the first.
            layer.allow_backend_reduction = False
            self.assertEqual(
                unmet_tpu_nvfp4_fused_rs_requirements(
                    layer, group_size=8, no_combine=True
                ),
                [
                    f"group_size in {TPU_NVFP4_SUPPORTED_GROUP_SIZES}",
                    "layer allows backend reduction",
                    "no_combine matches the model's post-expert-norm contract",
                ],
            )

    def test_non_rs_no_combine_passes_masked_routing_to_pallas(self):
        from sglang_tpu import fp4_moe

        hidden_states = torch.randn(2, HIDDEN_SIZE, dtype=torch.bfloat16)
        topk_weights = torch.tensor([[0.75, 0.25], [0.60, 0.40]], dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 2], [3, 1]], dtype=torch.int64)
        dispatch_output = types.SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=(topk_weights, topk_ids, None),
        )
        layer = _make_block512_layer(include_unused_params=False)
        layer.num_local_experts = 2
        layer.moe_ep_rank = 0
        fused_moe_call = mock.Mock(
            return_value=torch.empty(2, 2, HIDDEN_SIZE, dtype=torch.bfloat16)
        )

        with mock.patch.object(fp4_moe, "fused_moe_call", fused_moe_call):
            result = fp4_moe.apply_tpu_nvfp4_moe(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
                apply_router_weight_on_input=False,
                no_combine=True,
            )

        call = fused_moe_call.call_args.kwargs
        self.assertTrue(call["no_combine"])
        torch.testing.assert_close(
            call["topk_weights"],
            torch.tensor([[0.75, 0.0], [0.0, 0.40]], dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            call["topk_ids"],
            torch.tensor([[0, -1], [-1, 1]], dtype=torch.int32),
        )
        self.assertEqual(tuple(result.hidden_states.shape), (2, 2, HIDDEN_SIZE))

    def test_fused_backend_prepares_layer_eagerly(self):
        from sglang_tpu import fp4_moe

        for group_size in SUPPORTED_BLOCK_SIZES:
            with self.subTest(group_size=group_size):
                layer = _make_block512_layer(group_size=group_size)
                prepare_fused_moe_rs = mock.Mock()
                fused_rs_module = types.SimpleNamespace(
                    prepare_fused_moe_rs=prepare_fused_moe_rs
                )

                with (
                    mock.patch.dict(
                        sys.modules,
                        {"sglang_tpu.moe.pallas.fused_moe_rs_op": fused_rs_module},
                    ),
                    mock.patch.object(
                        fp4_moe,
                        "_materialize_packed_fp4",
                        side_effect=lambda value: value,
                    ),
                ):
                    fp4_moe.prepare_tpu_nvfp4_moe_weights(
                        layer, group_size=group_size, use_fused_rs=True
                    )

                prepare_fused_moe_rs.assert_called_once_with(layer)
                # A layer that does not declare the shared post-expert norm
                # contract still gets the cached flag, so apply never needs
                # getattr.
                self.assertFalse(layer._tpu_fused_rs_has_post_expert_norm)
                self.assertEqual(
                    tuple(layer.w13_weight.shape),
                    (EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
                )
                # Fused-RS takes compact [E, K_blocks, N] scales (no unit dim).
                self.assertEqual(
                    tuple(layer.w13_weight_scale.shape),
                    (EXPERTS, HIDDEN_SIZE // group_size, 2 * INTERMEDIATE_SIZE),
                )
                self.assertEqual(
                    tuple(layer.w2_weight_scale.shape),
                    (EXPERTS, INTERMEDIATE_SIZE // group_size, HIDDEN_SIZE),
                )

    def test_fused_prepare_supplies_every_attribute_apply_reads(self):
        """Producer -> consumer contract for the shared fused-RS attributes.

        The apply tests hand-set these attributes on their own fixtures and the
        eager-prepare test replaces the producer with a bare Mock, so neither
        side of that boundary pins the contract. Here prepare and apply run
        against the SAME layer object with only the kernel entry point mocked,
        so anything apply reads must have been written by prepare.
        """
        from sglang_tpu import fp4_moe

        layer = _make_block512_layer()
        layer.moe_ep_size = 1

        def fake_prepare_fused_moe_rs(prepared_layer):
            # Mirrors fused_moe_rs_op.prepare_fused_moe_rs, which cannot run
            # here (it needs an initialized TP group). Keep the attribute names
            # in sync with that function.
            prepared_layer._tpu_fused_rs_group_offset = torch.tensor(
                [0], dtype=torch.int32
            )

        hidden_states = torch.randn(3, HIDDEN_SIZE, dtype=torch.bfloat16)
        dispatch_output = types.SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=(
                torch.rand(3, 2, dtype=torch.bfloat16),
                torch.tensor([[0, 1], [1, -1], [0, 1]], dtype=torch.int64),
                None,
            ),
        )
        fused_moe_rs_gather = mock.Mock(
            return_value=torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)
        )
        fused_rs_module = types.SimpleNamespace(
            prepare_fused_moe_rs=fake_prepare_fused_moe_rs,
            fused_moe_rs_gather=fused_moe_rs_gather,
        )

        with (
            mock.patch.dict(
                sys.modules,
                {"sglang_tpu.moe.pallas.fused_moe_rs_op": fused_rs_module},
            ),
            mock.patch.object(
                fp4_moe, "_materialize_packed_fp4", side_effect=lambda value: value
            ),
        ):
            fp4_moe.prepare_tpu_nvfp4_moe_weights(layer, use_fused_rs=True)
            fp4_moe.apply_tpu_nvfp4_moe(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
                apply_router_weight_on_input=False,
                use_fused_rs=True,
            )

        # The prepared group offset must reach the kernel unchanged.
        call = fused_moe_rs_gather.call_args.kwargs
        self.assertNotIn("collective_id", call)
        self.assertIs(call["group_offset"], layer._tpu_fused_rs_group_offset)
        self.assertEqual(call["world_size"], layer.moe_ep_size)
        self.assertFalse(call["has_post_expert_norm"])

    def test_fused_apply_uses_shared_gather_without_fp8_payload(self):
        from sglang_tpu import fp4_moe

        hidden_states = torch.randn(3, HIDDEN_SIZE, dtype=torch.bfloat16)
        topk_weights = torch.rand(3, 2, dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [1, -1], [0, 1]], dtype=torch.int64)
        dispatch_output = types.SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=(topk_weights, topk_ids, None),
        )
        layer = _make_block512_layer(include_unused_params=False)
        layer._tpu_fused_rs_has_post_expert_norm = False
        layer._tpu_fused_rs_group_offset = torch.tensor([0], dtype=torch.int32)
        layer.moe_ep_size = 1
        from sglang_tpu.moe.pallas import fused_moe_rs_op

        # Autospec is intentional: it catches adapter/backend API skew during a
        # rebase instead of letting a permissive Mock accept removed keywords.
        fused_moe_rs_gather = mock.create_autospec(
            fused_moe_rs_op.fused_moe_rs_gather,
            return_value=torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16),
        )

        with mock.patch.object(
            fused_moe_rs_op, "fused_moe_rs_gather", fused_moe_rs_gather
        ):
            result = fp4_moe.apply_tpu_nvfp4_moe(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
                apply_router_weight_on_input=False,
                use_fused_rs=True,
            )

        call = fused_moe_rs_gather.call_args.kwargs
        self.assertFalse(call["fp8_direct_write"])
        self.assertFalse(call["has_post_expert_norm"])
        self.assertNotIn("collective_id", call)
        self.assertEqual(call["topk_ids"][1, 1].item(), 0)
        self.assertEqual(call["topk_weights"][1, 1].item(), 0)
        self.assertEqual(tuple(result.hidden_states.shape), (3, HIDDEN_SIZE))

    def test_fused_apply_runs_8k_once_and_chunks_only_above_8k(self):
        from sglang_tpu import fp4_moe

        for num_tokens, expected_call_rows in (
            (8192, [8192]),
            (8193, [8192, 1]),
        ):
            with self.subTest(num_tokens=num_tokens):
                hidden_states = torch.randn(
                    num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16
                )
                topk_weights = torch.rand(num_tokens, 2, dtype=torch.bfloat16)
                topk_ids = torch.zeros(num_tokens, 2, dtype=torch.int64)
                dispatch_output = types.SimpleNamespace(
                    hidden_states=hidden_states,
                    topk_output=(topk_weights, topk_ids, None),
                )
                layer = _make_block512_layer(include_unused_params=False)
                layer._tpu_fused_rs_has_post_expert_norm = False
                layer._tpu_fused_rs_group_offset = torch.tensor([0], dtype=torch.int32)
                layer.moe_ep_size = 1
                fused_moe_rs_gather = mock.Mock(
                    side_effect=lambda **kwargs: kwargs["hidden_states"]
                )
                fused_rs_module = types.SimpleNamespace(
                    fused_moe_rs_gather=fused_moe_rs_gather
                )

                with mock.patch.dict(
                    sys.modules,
                    {"sglang_tpu.moe.pallas.fused_moe_rs_op": fused_rs_module},
                ):
                    result = fp4_moe.apply_tpu_nvfp4_moe(
                        layer=layer,
                        dispatch_output=dispatch_output,
                        activation="silu",
                        apply_router_weight_on_input=False,
                        use_fused_rs=True,
                    )

                self.assertEqual(
                    fused_moe_rs_gather.call_count, len(expected_call_rows)
                )
                self.assertEqual(
                    [
                        call.kwargs["hidden_states"].shape[0]
                        for call in fused_moe_rs_gather.call_args_list
                    ],
                    expected_call_rows,
                )
                # The shared backend owns collective-ID reuse; the adapter only
                # issues the same deterministic chunk sequence on every rank.
                self.assertTrue(
                    all(
                        "collective_id" not in call.kwargs
                        for call in fused_moe_rs_gather.call_args_list
                    )
                )
                torch.testing.assert_close(result.hidden_states, hidden_states)

    def test_fused_apply_chunks_sequence_parallel_inputs_by_local_row(self):
        from sglang_tpu import fp4_moe

        rows_per_partition = 4097
        partition_count = 2
        num_tokens = rows_per_partition * partition_count
        # BF16 exactly represents these small repeating IDs, which makes the
        # partition/chunk order observable without relying on large integers.
        row_ids = (torch.arange(num_tokens, dtype=torch.int32) % 251).to(torch.bfloat16)
        hidden_states = row_ids[:, None].expand(-1, HIDDEN_SIZE).contiguous()
        topk_weights = torch.ones(num_tokens, 2, dtype=torch.bfloat16)
        topk_ids = torch.zeros(num_tokens, 2, dtype=torch.int64)
        dispatch_output = types.SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=(topk_weights, topk_ids, None),
        )
        prepared_input = types.SimpleNamespace(
            dispatch_output=dispatch_output,
            plan=types.SimpleNamespace(keep_output_shard=True),
            output_token_count=rows_per_partition,
        )
        layer = _make_block512_layer(include_unused_params=False)
        layer._tpu_fused_rs_has_post_expert_norm = False
        layer._tpu_fused_rs_group_offset = torch.tensor([0], dtype=torch.int32)
        layer.moe_ep_size = 2

        # Model rank 0's output is the first partition of each kernel result.
        fused_moe_rs_gather = mock.Mock(
            side_effect=lambda **kwargs: kwargs["hidden_states"]
        )
        fused_rs_module = types.SimpleNamespace(fused_moe_rs_gather=fused_moe_rs_gather)
        with (
            mock.patch.dict(
                sys.modules,
                {"sglang_tpu.moe.pallas.fused_moe_rs_op": fused_rs_module},
            ),
            mock.patch.object(
                fp4_moe,
                "prepare_fused_moe_sequence_parallel_input",
                return_value=prepared_input,
            ),
        ):
            result = fp4_moe.apply_tpu_nvfp4_moe(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
                apply_router_weight_on_input=False,
                use_fused_rs=True,
            )

        self.assertEqual(
            [
                call.kwargs["hidden_states"].shape[0]
                for call in fused_moe_rs_gather.call_args_list
            ],
            [8192, 2],
        )
        first_call_rows = fused_moe_rs_gather.call_args_list[0].kwargs["hidden_states"][
            :, 0
        ]
        torch.testing.assert_close(
            first_call_rows,
            torch.cat(
                (
                    row_ids[:4096],
                    row_ids[rows_per_partition : rows_per_partition + 4096],
                )
            ),
        )
        second_call_rows = fused_moe_rs_gather.call_args_list[1].kwargs[
            "hidden_states"
        ][:, 0]
        torch.testing.assert_close(
            second_call_rows,
            torch.stack((row_ids[4096], row_ids[8193])),
        )
        torch.testing.assert_close(
            result.hidden_states[:, 0], row_ids[:rows_per_partition]
        )

    def test_fused_backend_caches_declared_post_expert_norm_weight(self):
        from sglang_tpu import fp4_moe

        layer = _make_block512_layer(include_unused_params=False)
        layer.hidden_size = HIDDEN_SIZE
        norm_weight = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16)
        layer.supports_tpu_fused_rs_post_expert_norm = True
        layer.tpu_fused_rs_post_expert_norm_weight = lambda: norm_weight
        fused_rs_module = types.SimpleNamespace(prepare_fused_moe_rs=mock.Mock())

        with (
            mock.patch.dict(
                sys.modules,
                {"sglang_tpu.moe.pallas.fused_moe_rs_op": fused_rs_module},
            ),
            mock.patch.object(
                fp4_moe, "_materialize_packed_fp4", side_effect=lambda value: value
            ),
        ):
            fp4_moe.prepare_tpu_nvfp4_moe_weights(layer, use_fused_rs=True)

        self.assertTrue(layer._tpu_fused_rs_has_post_expert_norm)
        self.assertIs(layer._tpu_fused_rs_post_expert_norm_weight, norm_weight)
        # The alias must not become a second registered checkpoint parameter.
        self.assertNotIn("_tpu_fused_rs_post_expert_norm_weight", layer._parameters)

    def test_fused_backend_rejects_missing_post_expert_norm_weight_hook(self):
        from sglang_tpu import fp4_moe

        layer = _make_block512_layer(include_unused_params=False)
        layer.hidden_size = HIDDEN_SIZE
        # Capability declared but the weight hook is missing.
        layer.supports_tpu_fused_rs_post_expert_norm = True
        fused_rs_module = types.SimpleNamespace(prepare_fused_moe_rs=mock.Mock())

        with (
            mock.patch.dict(
                sys.modules,
                {"sglang_tpu.moe.pallas.fused_moe_rs_op": fused_rs_module},
            ),
            mock.patch.object(
                fp4_moe, "_materialize_packed_fp4", side_effect=lambda value: value
            ),
            self.assertRaisesRegex(TypeError, "post-expert norm requires"),
        ):
            fp4_moe.prepare_tpu_nvfp4_moe_weights(layer, use_fused_rs=True)

    def test_fused_rs_group_offset_stays_model_sharded(self):
        repo_root = pathlib.Path(__file__).resolve().parents[5]
        wrapper = (repo_root / "sglang_tpu/moe/pallas/fused_moe_rs_op.py").read_text()
        body = (repo_root / "sglang_tpu/moe/pallas/fused_rs/fused_moe.py").read_text()

        self.assertIn("group_offset_input=group_offset", wrapper)
        self.assertIn("weight_p_spec,  # group_offset", body)

    def test_fused_packed_transpose_preserves_codes(self):
        from sglang_tpu.fp4_moe import (
            _transpose_packed_fp4,
        )

        codes = torch.arange(2 * 4 * 6, dtype=torch.uint8).reshape(2, 4, 6) & 0x0F
        packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
        transposed = _transpose_packed_fp4(packed)
        unpacked = torch.stack((transposed & 0x0F, transposed >> 4), dim=-1).reshape(
            2, 6, 4
        )

        torch.testing.assert_close(unpacked, codes.transpose(1, 2))

    def test_fused_rs_selects_expected_activation_mode_by_group_size(self):
        import jax.numpy as jnp

        from sglang_tpu.moe.pallas.fused_rs.gmm_v2_gather_scatter import (
            get_maybe_quantize_lhs,
        )

        cases = (
            (128, 256, False),  # W4A16
            (256, 256, True),  # W4A8
            (512, 512, True),  # W4A8
        )
        for rhs_group_size, lhs_group_size, expected in cases:
            with self.subTest(group_size=rhs_group_size):
                self.assertEqual(
                    get_maybe_quantize_lhs(
                        jnp.float4_e2m1fn,
                        rhs_group_size,
                        lhs_group_size,
                    ),
                    expected,
                )


class TestNvfp4ModelOptTpuGuard(CustomTestCase):
    """ModelOpt FP4 backend contracts that are reachable without JAX."""

    @staticmethod
    def _checkpoint_config(**overrides):
        quantization = {
            "quant_algo": "NVFP4",
            "group_size": BLOCK_SIZE,
            "exclude_modules": ["self_attn", "lm_head"],
        }
        quantization.update(overrides)
        return {"quant_method": "modelopt", "quantization": quantization}

    def test_tpu_platform_factory_selects_tpu_config(self):
        from sglang.srt.layers import quantization
        from sglang_tpu.quantization import modelopt_fp4
        from sglang_tpu.tpu_platform import TpuSRTPlatform

        platform = TpuSRTPlatform()
        self.assertIs(
            platform.get_quantization_config("modelopt_fp4"),
            modelopt_fp4.TpuModelOptCompatibleFp4Config,
        )
        mixed_config_cls = platform.get_quantization_config("modelopt_mixed")
        self.assertIs(mixed_config_cls, modelopt_fp4.TpuModelOptMixedPrecisionConfig)
        with self.assertRaisesRegex(NotImplementedError, "not supported on TPU"):
            mixed_config_cls.from_config({})
        self.assertIsNone(platform.get_quantization_config("modelopt_fp8"))
        with (
            mock.patch.dict("os.environ", {"SGLANG_USE_CPU_ENGINE": "1"}),
            mock.patch.object(quantization, "current_platform", platform),
            mock.patch.object(quantization, "is_cpu", return_value=False) as is_cpu,
        ):
            self.assertIs(
                quantization.get_quantization_config("modelopt_fp4"),
                modelopt_fp4.TpuModelOptCompatibleFp4Config,
            )
            self.assertIs(
                quantization.get_quantization_config("modelopt_mixed"),
                modelopt_fp4.TpuModelOptMixedPrecisionConfig,
            )
        self.assertEqual(is_cpu.call_count, 2)

    def test_hybrid_loader_preserves_metadata_and_uses_platform_config(self):
        from sglang.srt.layers.quantization import modelopt_quant
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.model_loader import loader
        from sglang_tpu.quantization import modelopt_fp4

        fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]},
        )
        model_config = types.SimpleNamespace(
            quantization="fp8",
            is_fp4_experts=False,
            hf_config=types.SimpleNamespace(architectures=[]),
            nvfp4_moe_meta={
                "group_size": BLOCK_SIZE,
                "exclude_modules": ["vision"],
                "global_scale": False,
                "scale_dtype": "float32",
            },
            dtype=torch.bfloat16,
        )
        model_cls = type(
            "HybridModel",
            (),
            {"packed_modules_mapping": fp8_config.packed_modules_mapping},
        )
        with (
            mock.patch.object(
                loader, "get_model_architecture", return_value=(model_cls, None)
            ),
            mock.patch.object(loader, "get_quant_config", return_value=fp8_config),
            mock.patch.object(
                loader, "get_device_capability", return_value=(None, None)
            ),
            mock.patch(
                "sglang.srt.layers.quantization.get_quantization_config",
                return_value=modelopt_fp4.TpuModelOptCompatibleFp4Config,
            ) as get_quantization_config,
        ):
            config = loader._get_quantization_config(
                model_config, types.SimpleNamespace()
            )

        self.assertIsInstance(config, modelopt_quant.HybridFp8NvFp4Config)
        self.assertIsInstance(
            config.nvfp4_config, modelopt_fp4.TpuModelOptCompatibleFp4Config
        )
        self.assertEqual(config.nvfp4_config.group_size, BLOCK_SIZE)
        self.assertIs(config.nvfp4_config.global_scale, False)
        self.assertEqual(config.nvfp4_config.scale_dtype, "float32")
        self.assertEqual(
            config.nvfp4_config.exclude_modules,
            # "stages.*" joined the hybrid loader's exclude list in the
            # upstream sync (staged-decoder archs are never FP4-quantized).
            ["vision", "model.decoder.*", "stages.*"],
        )
        get_quantization_config.assert_called_once_with("modelopt_fp4")

    def test_shared_modelopt_config_preserves_extension_metadata_and_dispatch(self):
        from sglang.srt.layers.quantization import modelopt_quant

        config = modelopt_quant.ModelOptFp4Config.from_config(
            self._checkpoint_config(global_scale=False, scale_dtype="fp32")
        )
        self.assertIs(type(config), modelopt_quant.ModelOptFp4Config)
        self.assertTrue(config.is_checkpoint_nvfp4_serialized)
        self.assertEqual(config.kv_cache_quant_algo, "auto")
        self.assertEqual(config.group_size, BLOCK_SIZE)
        self.assertEqual(config.exclude_modules, ["self_attn", "lm_head"])
        self.assertIs(config.global_scale, False)
        self.assertEqual(config.scale_dtype, "fp32")
        with mock.patch.object(config, "_get_quant_method") as get_quant_method:
            config.get_quant_method(torch.nn.Module(), "model.layers.0.mlp")
        self.assertIs(
            get_quant_method.call_args.kwargs["Linear"],
            modelopt_quant.ModelOptFp4LinearMethod,
        )
        self.assertIs(
            get_quant_method.call_args.kwargs["Moe"],
            modelopt_quant.ModelOptNvFp4FusedMoEMethod,
        )

    def test_shared_modelopt_config_preserves_flat_metadata_without_normalizing(self):
        from sglang.srt.layers.quantization import modelopt_quant

        config = modelopt_quant.ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": BLOCK_SIZE,
                "ignore": ["self_attn", "lm_head"],
                "global_scale": False,
                "scale_dtype": " BF16 ",
            }
        )

        self.assertIs(config.global_scale, False)
        self.assertEqual(config.scale_dtype, " BF16 ")

    def test_standard_nvfp4_methods_accept_standard_defaults(self):
        from sglang.srt.layers.quantization import modelopt_quant

        config = modelopt_quant.ModelOptFp4Config.from_config(
            self._checkpoint_config(group_size=16)
        )
        self.assertIs(config.global_scale, True)
        self.assertIsNone(config.scale_dtype)

        self.assertIsInstance(
            modelopt_quant.ModelOptFp4LinearMethod(config),
            modelopt_quant.ModelOptFp4LinearMethod,
        )
        self.assertIsInstance(
            modelopt_quant.ModelOptNvFp4A16LinearMethod(config),
            modelopt_quant.ModelOptNvFp4A16LinearMethod,
        )
        backend = mock.MagicMock()
        backend.is_auto.return_value = True
        backend.is_flashinfer_trtllm.return_value = False
        backend.is_flashinfer_trtllm_routed.return_value = False
        with (
            mock.patch.object(
                modelopt_quant, "get_moe_runner_backend", return_value=backend
            ),
            mock.patch.object(modelopt_quant, "is_cuda", return_value=False),
            mock.patch.object(
                modelopt_quant,
                "get_platform",
                return_value=mock.MagicMock(is_blackwell=True),
            ),
        ):
            self.assertIsInstance(
                modelopt_quant.ModelOptNvFp4FusedMoEMethod(config),
                modelopt_quant.ModelOptNvFp4FusedMoEMethod,
            )

    def test_standard_nvfp4_methods_reject_each_extension_axis(self):
        from sglang.srt.layers.quantization import modelopt_quant

        method_classes = (
            modelopt_quant.ModelOptFp4LinearMethod,
            modelopt_quant.ModelOptNvFp4A16LinearMethod,
            modelopt_quant.ModelOptNvFp4FusedMoEMethod,
        )
        cases = [
            ("block size", {"group_size": BLOCK_SIZE}, "group_size"),
            (
                "folded global scale",
                {"group_size": 16, "global_scale": False},
                "global_scale",
            ),
            *[
                (
                    f"declared {scale_dtype} scales",
                    {"group_size": 16, "scale_dtype": scale_dtype},
                    "scale_dtype",
                )
                for scale_dtype in (
                    "float32",
                    "fp32",
                    "bfloat16",
                    "bf16",
                    "float8_e4m3fn",
                    "fp8",
                )
            ],
            (
                "combined extension",
                {
                    "group_size": BLOCK_SIZE,
                    "global_scale": False,
                    "scale_dtype": "fp32",
                },
                "group_size",
            ),
        ]
        for case_name, config_kwargs, rejected_field in cases:
            config = modelopt_quant.ModelOptFp4Config(**config_kwargs)
            for method_cls in method_classes:
                with self.subTest(case=case_name, method=method_cls.__name__):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"large-block extension.*{rejected_field}",
                    ):
                        method_cls(config)

    def test_tpu_config_selects_moe_adapter_and_rejects_dense_fp4(self):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang_tpu.quantization import modelopt_fp4

        config = modelopt_fp4.TpuModelOptCompatibleFp4Config.from_config(
            self._checkpoint_config()
        )
        method = config.get_quant_method(
            mock.Mock(spec=FusedMoE), "model.layers.0.mlp.experts"
        )

        self.assertIsInstance(method, modelopt_fp4.TpuModelOptNvFp4FusedMoEMethod)
        self.assertNotIn("handles_moe_reduction", method.__dict__)
        self.assertFalse(method.handles_moe_reduction)
        dense_layer = mock.Mock(spec=LinearBase)
        with self.assertRaisesRegex(NotImplementedError, "dense layer"):
            config.get_quant_method(dense_layer, "model.layers.0.mlp.shared_expert")
        self.assertIsInstance(
            config.get_quant_method(dense_layer, "model.layers.0.self_attn.q_proj"),
            UnquantizedLinearMethod,
        )
        self.assertIsNone(
            config.get_quant_method(
                mock.Mock(spec=FusedMoE), "model.layers.0.self_attn.experts"
            )
        )

    def test_hybrid_nvfp4_dispatches_through_tpu_config(self):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization import modelopt_quant
        from sglang_tpu.quantization import modelopt_fp4

        tpu_config = modelopt_fp4.TpuModelOptCompatibleFp4Config.from_config(
            self._checkpoint_config()
        )
        hybrid_config = modelopt_quant.HybridFp8NvFp4Config(
            fp8_config=types.SimpleNamespace(is_fp4_experts=False),
            nvfp4_config=tpu_config,
        )

        method = hybrid_config.get_quant_method(
            mock.Mock(spec=FusedMoE), "model.layers.0.mlp.experts"
        )
        self.assertIsInstance(method, modelopt_fp4.TpuModelOptNvFp4FusedMoEMethod)

        unsupported_tpu_config = (
            modelopt_fp4.TpuModelOptCompatibleFp4Config.from_config(
                self._checkpoint_config(group_size=8)
            )
        )
        unsupported_hybrid = modelopt_quant.HybridFp8NvFp4Config(
            fp8_config=types.SimpleNamespace(is_fp4_experts=False),
            nvfp4_config=unsupported_tpu_config,
        )
        with self.assertRaisesRegex(NotImplementedError, "group_size in"):
            unsupported_hybrid.get_quant_method(
                mock.Mock(spec=FusedMoE), "model.layers.0.mlp.experts"
            )

    def test_tpu_config_parses_flat_checkpoint_metadata(self):
        from sglang_tpu.quantization import modelopt_fp4

        config = modelopt_fp4.TpuModelOptCompatibleFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": BLOCK_SIZE,
                "ignore": ["self_attn", "lm_head"],
                "global_scale": False,
                "scale_dtype": "BF16",
            }
        )

        self.assertIs(config.global_scale, False)
        self.assertEqual(config.scale_dtype, "BF16")
        self.assertEqual(config.exclude_modules, ["self_attn", "lm_head"])
        method = modelopt_fp4.TpuModelOptNvFp4FusedMoEMethod(config)
        self.assertEqual(method._weight_scale_dtype(), torch.bfloat16)
        self.assertEqual(method._prepared_scale_dtype, torch.bfloat16)

    def test_tpu_moe_accepts_and_honors_supported_scale_matrix(self):
        from sglang_tpu.quantization import modelopt_fp4

        dtype_aliases = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float8_e4m3fn": torch.float8_e4m3fn,
            "fp8": torch.float8_e4m3fn,
        }
        for global_scale in (True, False):
            for scale_dtype in (None, *dtype_aliases):
                overrides = {"global_scale": global_scale}
                if scale_dtype is not None:
                    overrides["scale_dtype"] = scale_dtype
                config = modelopt_fp4.TpuModelOptCompatibleFp4Config.from_config(
                    self._checkpoint_config(**overrides)
                )
                method = modelopt_fp4.TpuModelOptNvFp4FusedMoEMethod(config)
                expected_checkpoint_dtype = (
                    dtype_aliases[scale_dtype]
                    if scale_dtype is not None
                    else (torch.float8_e4m3fn if global_scale else torch.float32)
                )
                expected_prepared_dtype = (
                    dtype_aliases[scale_dtype]
                    if scale_dtype is not None
                    else (torch.bfloat16 if global_scale else torch.float32)
                )
                with self.subTest(global_scale=global_scale, scale_dtype=scale_dtype):
                    self.assertIs(config.global_scale, global_scale)
                    self.assertEqual(config.scale_dtype, scale_dtype)
                    self.assertEqual(
                        method._weight_scale_dtype(), expected_checkpoint_dtype
                    )
                    self.assertEqual(
                        method._prepared_scale_dtype, expected_prepared_dtype
                    )
                    self.assertIs(method._uses_global_scale_parameters(), global_scale)

    def test_tpu_moe_rejects_malformed_extension_metadata(self):
        from sglang_tpu.quantization import modelopt_fp4

        cases = (
            ({"global_scale": "false"}, "global_scale to be a bool"),
            ({"global_scale": 1}, "global_scale to be a bool"),
            ({"global_scale": None}, "global_scale to be a bool"),
            ({"scale_dtype": 32}, "scale_dtype to be a string"),
            ({"scale_dtype": ""}, "Unsupported.*scale_dtype"),
            ({"scale_dtype": "float16"}, "Unsupported.*scale_dtype"),
        )
        for overrides, error in cases:
            config = modelopt_fp4.TpuModelOptCompatibleFp4Config.from_config(
                self._checkpoint_config(**overrides)
            )
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, error):
                    modelopt_fp4.TpuModelOptNvFp4FusedMoEMethod(config)

    def test_tpu_moe_accepts_supported_group_sizes(self):
        from sglang_tpu.quantization import modelopt_fp4

        for group_size in SUPPORTED_BLOCK_SIZES:
            config = modelopt_fp4.TpuModelOptCompatibleFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                group_size=group_size,
                exclude_modules=[],
            )
            with self.subTest(group_size=group_size):
                method = modelopt_fp4.TpuModelOptNvFp4FusedMoEMethod(config)
                self.assertEqual(method.quant_config.group_size, group_size)

    def test_tpu_moe_rejects_unsupported_group_sizes(self):
        from sglang_tpu.quantization import modelopt_fp4

        for group_size in (8, 96, 1024, None):
            config = modelopt_fp4.TpuModelOptCompatibleFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                group_size=group_size,
                exclude_modules=[],
            )
            with (
                self.subTest(group_size=group_size),
                self.assertRaisesRegex(NotImplementedError, "group_size in"),
            ):
                modelopt_fp4.TpuModelOptNvFp4FusedMoEMethod(config)

    def test_folded_scale_checkpoint_skips_unused_auxiliary_parameters(self):
        from sglang_tpu.quantization import modelopt_fp4

        config = modelopt_fp4.TpuModelOptCompatibleFp4Config.from_config(
            self._checkpoint_config(global_scale=False, scale_dtype="fp32")
        )
        self.assertIs(config.global_scale, False)
        self.assertEqual(config.scale_dtype, "fp32")
        layer = torch.nn.Module()
        layer.num_local_experts = EXPERTS
        layer.num_experts = EXPERTS
        layer.moe_runner_config = types.SimpleNamespace(is_gated=True)

        method = modelopt_fp4.TpuModelOptNvFp4FusedMoEMethod(config)
        method.create_weights(
            layer,
            num_experts=EXPERTS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size_per_partition=INTERMEDIATE_SIZE,
            params_dtype=torch.bfloat16,
        )

        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            self.assertIn(name, layer._parameters)
        for name in (
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_input_scale",
            "w2_input_scale",
        ):
            self.assertNotIn(name, layer._parameters)


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
class TestNvfp4FusedRsGateTiming(CustomTestCase):
    """The fused-RS decision must outlive `FusedMoE.__init__`.

    `create_moe_runner` runs from the constructor, before models assign
    `allow_backend_reduction` or declare a post-expert norm. Deciding there
    reads the constructor defaults and pins fused-RS off for the whole run --
    silently, since the stock path still serves correct numerics.
    """

    def _run_gate(
        self,
        *,
        ep_size: int = 4,
        world_size: int = 4,
        model_contract: bool = False,
        backend: str = "auto",
        group_size: int = BLOCK_SIZE,
    ):
        from sglang.srt.layers.moe import MoeRunnerBackend
        from sglang_tpu.quantization import modelopt_fp4

        config = modelopt_fp4.TpuModelOptCompatibleFp4Config.from_config(
            {
                "quant_method": "modelopt",
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": group_size,
                    "exclude_modules": ["self_attn", "lm_head"],
                },
            }
        )
        moe_runner_config = types.SimpleNamespace(
            is_gated=True,
            activation="silu",
            apply_router_weight_on_input=False,
            no_combine=False,
        )
        # The model has not applied its overrides yet at construction time.
        layer = types.SimpleNamespace(
            moe_ep_size=ep_size,
            allow_backend_reduction=False,
            num_fused_shared_experts=0,
            moe_runner_config=moe_runner_config,
        )
        if model_contract:
            layer.supports_tpu_fused_rs_post_expert_norm = True
            moe_runner_config.no_combine = True
        prepared = {}

        def _fake_prepare(layer, **kwargs):
            prepared.update(kwargs)

        with (
            mock.patch.object(
                modelopt_fp4,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend(backend),
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=types.SimpleNamespace(
                    attn_dp_size=1,
                    enable_dp_attention=False,
                    attn_dp_enabled=False,
                    tp_group=types.SimpleNamespace(world_size=world_size),
                    world_group=types.SimpleNamespace(world_size=world_size),
                ),
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=types.SimpleNamespace(
                    moe=types.SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
                ),
            ),
            mock.patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=types.SimpleNamespace(is_none=lambda: True),
            ),
            mock.patch(
                "sglang_tpu.fp4_moe.prepare_tpu_nvfp4_moe_weights",
                _fake_prepare,
            ),
        ):
            method = modelopt_fp4.TpuModelOptNvFp4FusedMoEMethod(config)
            method.create_moe_runner(layer, moe_runner_config)
            constructed = method.handles_moe_reduction
            if not model_contract:
                layer.allow_backend_reduction = True
            method.process_weights_after_loading(layer)
        return constructed, method.handles_moe_reduction, prepared, layer

    def test_gate_sees_overrides_applied_after_construction(self):
        for group_size in SUPPORTED_BLOCK_SIZES:
            with self.subTest(group_size=group_size):
                constructed, resolved, prepared, _ = self._run_gate(
                    group_size=group_size
                )

                self.assertFalse(
                    constructed, "gate must not be decided in the constructor"
                )
                self.assertTrue(resolved)
                self.assertTrue(prepared["use_fused_rs"])

    def test_gate_accepts_post_expert_norm_contract(self):
        _, resolved, prepared, layer = self._run_gate(model_contract=True)

        self.assertTrue(resolved)
        self.assertTrue(prepared["use_fused_rs"])
        self.assertTrue(layer.allow_backend_reduction)

    def test_explicit_pallas_backend_selects_fused_rs(self):
        _, resolved, prepared, _ = self._run_gate(backend="pallas_tpu_fused_rs")

        self.assertTrue(resolved)
        self.assertTrue(prepared["use_fused_rs"])

    def test_explicit_legacy_pallas_backend_disables_fused_rs(self):
        _, resolved, prepared, _ = self._run_gate(backend="pallas_tpu")

        self.assertFalse(resolved)
        self.assertFalse(prepared["use_fused_rs"])

    def test_gate_still_declines_an_unsupported_layout(self):
        # EP 2 inside a world of 4 is the mixed TP/EP shape the requirement
        # list rejects; deferring the decision must not soften the gate.
        _, resolved, prepared, _ = self._run_gate(ep_size=2, world_size=4)

        self.assertFalse(resolved)
        self.assertFalse(prepared["use_fused_rs"])

    def test_sub512_auto_backend_fails_closed_on_unsupported_layout(self):
        for group_size in (16, 32, 64, 128, 256):
            with (
                self.subTest(group_size=group_size),
                self.assertRaisesRegex(ValueError, "requires the fused-RS"),
            ):
                self._run_gate(
                    ep_size=2,
                    world_size=4,
                    group_size=group_size,
                )

    def test_explicit_pallas_backend_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unmet requirements"):
            self._run_gate(
                ep_size=2,
                world_size=4,
                backend="pallas_tpu_fused_rs",
            )

    def test_rejects_non_pallas_backend_on_tpu(self):
        with self.assertRaisesRegex(ValueError, "supports only"):
            self._run_gate(backend="triton")


if __name__ == "__main__":
    unittest.main()
