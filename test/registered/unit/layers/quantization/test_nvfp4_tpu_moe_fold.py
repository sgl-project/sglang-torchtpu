"""CPU tests for the TPU block-512 NVFP4 MoE weight contract."""

import sys
import types
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

BLOCK_SIZE = 512
EXPERTS = 2
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 512


def _make_block512_layer(include_unused_params: bool = True) -> torch.nn.Module:
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
                HIDDEN_SIZE // BLOCK_SIZE,
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
                INTERMEDIATE_SIZE // BLOCK_SIZE,
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


class TestNvfp4Block512Contract(CustomTestCase):
    def test_block512_layout_and_cleanup(self):
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
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

    def test_rejects_block16(self):
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        with self.assertRaisesRegex(NotImplementedError, "group_size=512"):
            prepare_tpu_nvfp4_moe_weights(_make_block512_layer(), group_size=16)

    def test_scale_storage_dtype(self):
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
            prepare_tpu_nvfp4_moe_weights,
        )

        layer = _make_block512_layer()
        prepare_tpu_nvfp4_moe_weights(layer, scale_dtype=torch.bfloat16)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.bfloat16)
        self.assertEqual(layer.w2_weight_scale.dtype, torch.bfloat16)

    def test_folds_non_unit_global_scales(self):
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
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
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
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
        parallel = types.SimpleNamespace(attn_dp_size=1)
        server_args = types.SimpleNamespace(
            enable_dp_attention=False,
            enable_fused_moe_sum_all_reduce=False,
        )
        a2a_backend = types.SimpleNamespace(is_none=lambda: True)
        with (
            mock.patch(
                "sglang.srt.distributed.get_world_group", return_value=world_group
            ),
            mock.patch("sglang.srt.distributed.get_tp_group", return_value=tp_group),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_server_args", return_value=server_args
            ),
            mock.patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=a2a_backend,
            ),
        ):
            self.assertTrue(
                can_use_tpu_nvfp4_fused_rs(
                    layer, group_size=BLOCK_SIZE, no_combine=False
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
            server_args.enable_dp_attention = True
            self.assertFalse(
                can_use_tpu_nvfp4_fused_rs(
                    layer, group_size=BLOCK_SIZE, no_combine=False
                )
            )

    def test_fused_backend_prepares_layer_eagerly(self):
        from sglang.srt.hardware_backend.tpu import fp4_moe

        layer = _make_block512_layer()
        prepare_fused_moe_rs = mock.Mock()
        fused_rs_module = types.SimpleNamespace(
            prepare_fused_moe_rs=prepare_fused_moe_rs
        )

        with (
            mock.patch.dict(
                sys.modules,
                {"sglang.srt.layers.moe.pallas.fused_moe_rs_op": fused_rs_module},
            ),
            mock.patch.object(
                fp4_moe, "_materialize_packed_fp4", side_effect=lambda value: value
            ),
        ):
            fp4_moe.prepare_tpu_nvfp4_moe_weights(layer, use_fused_rs=True)

        prepare_fused_moe_rs.assert_called_once_with(layer)
        # A layer that does not declare the shared post-expert norm contract
        # still gets the cached flag, so the apply path never needs getattr.
        self.assertFalse(layer._tpu_fused_rs_has_post_expert_norm)
        self.assertEqual(
            tuple(layer.w13_weight.shape),
            (EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        )

    def test_fused_prepare_supplies_every_attribute_apply_reads(self):
        """Producer -> consumer contract for the shared fused-RS attributes.

        The apply tests hand-set these attributes on their own fixtures and the
        eager-prepare test replaces the producer with a bare Mock, so neither
        side of that boundary pins the contract. That is exactly how a newly
        required ``collective_id`` reached the apply path with nothing in the
        suite writing it. Here prepare and apply run against the SAME layer
        object with only the kernel entry point mocked, so anything apply reads
        must have been written by prepare.
        """
        from sglang.srt.hardware_backend.tpu import fp4_moe

        layer = _make_block512_layer()
        layer.layer_id = 7

        def fake_prepare_fused_moe_rs(prepared_layer):
            # Mirrors fused_moe_rs_op.prepare_fused_moe_rs, which cannot run
            # here (it needs an initialized TP group). Keep the attribute names
            # and the collective-id formula in sync with that function.
            prepared_layer._tpu_fused_rs_physical_order = torch.tensor(
                [0], dtype=torch.int64
            )
            prepared_layer._tpu_fused_rs_group_offset = torch.tensor(
                [0, 0], dtype=torch.int32
            )
            prepared_layer._tpu_fused_rs_collective_id = 100 + int(
                prepared_layer.layer_id
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
                {"sglang.srt.layers.moe.pallas.fused_moe_rs_op": fused_rs_module},
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

        # Every layer-owned value the kernel receives came from prepare, not
        # from the fixture, so dropping or renaming one fails here.
        call = fused_moe_rs_gather.call_args.kwargs
        self.assertEqual(call["collective_id"], 100 + layer.layer_id)
        self.assertIs(call["group_offset"], layer._tpu_fused_rs_group_offset)
        self.assertIs(
            call["physical_order_to_tp_rank"], layer._tpu_fused_rs_physical_order
        )
        self.assertFalse(call["has_post_expert_norm"])

    def test_fused_apply_uses_shared_gather_without_fp8_payload(self):
        from sglang.srt.hardware_backend.tpu import fp4_moe

        hidden_states = torch.randn(3, HIDDEN_SIZE, dtype=torch.bfloat16)
        topk_weights = torch.rand(3, 2, dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, 1], [1, -1], [0, 1]], dtype=torch.int64)
        dispatch_output = types.SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=(topk_weights, topk_ids, None),
        )
        layer = _make_block512_layer(include_unused_params=False)
        layer._tpu_fused_rs_has_post_expert_norm = False
        layer._tpu_fused_rs_group_offset = torch.tensor([0, 0], dtype=torch.int32)
        layer._tpu_fused_rs_physical_order = torch.tensor([0], dtype=torch.int64)
        layer._tpu_fused_rs_collective_id = 107
        fused_moe_rs_gather = mock.Mock(
            return_value=torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)
        )
        fused_rs_module = types.SimpleNamespace(fused_moe_rs_gather=fused_moe_rs_gather)

        with mock.patch.dict(
            sys.modules,
            {"sglang.srt.layers.moe.pallas.fused_moe_rs_op": fused_rs_module},
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
        self.assertEqual(call["collective_id"], 107)
        self.assertEqual(call["topk_ids"][1, 1].item(), 0)
        self.assertEqual(call["topk_weights"][1, 1].item(), 0)
        self.assertEqual(tuple(result.hidden_states.shape), (3, HIDDEN_SIZE))

    def test_fused_apply_chunks_large_prefills(self):
        from sglang.srt.hardware_backend.tpu import fp4_moe

        num_tokens = 2050
        hidden_states = torch.randn(num_tokens, HIDDEN_SIZE, dtype=torch.bfloat16)
        topk_weights = torch.rand(num_tokens, 2, dtype=torch.bfloat16)
        topk_ids = torch.zeros(num_tokens, 2, dtype=torch.int64)
        dispatch_output = types.SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=(topk_weights, topk_ids, None),
        )
        layer = _make_block512_layer(include_unused_params=False)
        layer._tpu_fused_rs_has_post_expert_norm = False
        layer._tpu_fused_rs_group_offset = torch.tensor([0, 0], dtype=torch.int32)
        layer._tpu_fused_rs_physical_order = torch.tensor([0], dtype=torch.int64)
        layer._tpu_fused_rs_collective_id = 107
        fused_moe_rs_gather = mock.Mock(
            side_effect=lambda **kwargs: kwargs["hidden_states"]
        )
        fused_rs_module = types.SimpleNamespace(fused_moe_rs_gather=fused_moe_rs_gather)

        with mock.patch.dict(
            sys.modules,
            {"sglang.srt.layers.moe.pallas.fused_moe_rs_op": fused_rs_module},
        ):
            result = fp4_moe.apply_tpu_nvfp4_moe(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
                apply_router_weight_on_input=False,
                use_fused_rs=True,
            )

        self.assertEqual(fused_moe_rs_gather.call_count, 3)
        self.assertEqual(
            [
                call.kwargs["hidden_states"].shape[0]
                for call in fused_moe_rs_gather.call_args_list
            ],
            [1024, 1024, 2],
        )
        # The chunks are sequential invocations of the same layer's collective,
        # so they share its id; every rank issues the same chunk sequence.
        self.assertEqual(
            {
                call.kwargs["collective_id"]
                for call in fused_moe_rs_gather.call_args_list
            },
            {107},
        )
        torch.testing.assert_close(result.hidden_states, hidden_states)

    def test_fused_backend_caches_declared_post_expert_norm_weight(self):
        from sglang.srt.hardware_backend.tpu import fp4_moe

        layer = _make_block512_layer(include_unused_params=False)
        layer.hidden_size = HIDDEN_SIZE
        norm_weight = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16)
        layer.supports_tpu_fused_rs_post_expert_norm = True
        layer.tpu_fused_rs_post_expert_norm_weight = lambda: norm_weight
        layer.finalize_tpu_fused_rs_fallback = lambda output, dispatch_output: output
        fused_rs_module = types.SimpleNamespace(prepare_fused_moe_rs=mock.Mock())

        with (
            mock.patch.dict(
                sys.modules,
                {"sglang.srt.layers.moe.pallas.fused_moe_rs_op": fused_rs_module},
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

    def test_fused_backend_rejects_incomplete_post_expert_norm_contract(self):
        from sglang.srt.hardware_backend.tpu import fp4_moe

        layer = _make_block512_layer(include_unused_params=False)
        layer.hidden_size = HIDDEN_SIZE
        # Capability declared but the weight/fallback hooks are missing.
        layer.supports_tpu_fused_rs_post_expert_norm = True
        fused_rs_module = types.SimpleNamespace(prepare_fused_moe_rs=mock.Mock())

        with (
            mock.patch.dict(
                sys.modules,
                {"sglang.srt.layers.moe.pallas.fused_moe_rs_op": fused_rs_module},
            ),
            mock.patch.object(
                fp4_moe, "_materialize_packed_fp4", side_effect=lambda value: value
            ),
            self.assertRaisesRegex(TypeError, "post-expert norm requires"),
        ):
            fp4_moe.prepare_tpu_nvfp4_moe_weights(layer, use_fused_rs=True)

    def test_fused_packed_transpose_preserves_codes(self):
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
            _transpose_packed_fp4,
        )

        codes = torch.arange(2 * 4 * 6, dtype=torch.uint8).reshape(2, 4, 6) & 0x0F
        packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
        transposed = _transpose_packed_fp4(packed)
        unpacked = torch.stack((transposed & 0x0F, transposed >> 4), dim=-1).reshape(
            2, 6, 4
        )

        torch.testing.assert_close(unpacked, codes.transpose(1, 2))

    def test_modelopt_rejects_block16_on_tpu(self):
        from sglang.srt.layers.quantization import modelopt_quant

        config = modelopt_quant.ModelOptFp4Config.from_config(
            {
                "quant_method": "modelopt",
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "exclude_modules": ["self_attn", "lm_head"],
                },
            }
        )
        with mock.patch.object(modelopt_quant, "_is_tpu", True):
            method = modelopt_quant.ModelOptNvFp4FusedMoEMethod(config)
            with self.assertRaisesRegex(NotImplementedError, "block-512"):
                method.create_weights(
                    torch.nn.Module(),
                    num_experts=EXPERTS,
                    hidden_size=HIDDEN_SIZE,
                    intermediate_size_per_partition=INTERMEDIATE_SIZE,
                    params_dtype=torch.bfloat16,
                )


if __name__ == "__main__":
    unittest.main()
