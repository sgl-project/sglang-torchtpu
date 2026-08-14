"""Tests for the TPU block-512 NVFP4 MoE weight contract.

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
register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")

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


@unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
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
        parallel = types.SimpleNamespace(attn_dp_size=1, enable_dp_attention=False)
        exec_ns = types.SimpleNamespace(
            moe=types.SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
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
            mock.patch("sglang.srt.runtime_context.get_exec", return_value=exec_ns),
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
            parallel.enable_dp_attention = True
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
        from sglang.srt.hardware_backend.tpu.fp4_moe import (
            TPU_NVFP4_GROUP_SIZE,
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
        parallel = types.SimpleNamespace(attn_dp_size=1, enable_dp_attention=False)
        exec_ns = types.SimpleNamespace(
            moe=types.SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
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
                    layer, group_size=16, no_combine=True
                ),
                [
                    f"group_size == {TPU_NVFP4_GROUP_SIZE}",
                    "layer allows backend reduction",
                    "no_combine matches the model's post-expert-norm contract",
                ],
            )

    def test_non_rs_no_combine_passes_masked_routing_to_pallas(self):
        from sglang.srt.hardware_backend.tpu import fp4_moe

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
            torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
        )
        self.assertEqual(tuple(result.hidden_states.shape), (2, 2, HIDDEN_SIZE))

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
        side of that boundary pins the contract. Here prepare and apply run
        against the SAME layer object with only the kernel entry point mocked,
        so anything apply reads must have been written by prepare.
        """
        from sglang.srt.hardware_backend.tpu import fp4_moe

        layer = _make_block512_layer()

        def fake_prepare_fused_moe_rs(prepared_layer):
            # Mirrors fused_moe_rs_op.prepare_fused_moe_rs, which cannot run
            # here (it needs an initialized TP group). Keep the attribute names
            # in sync with that function.
            prepared_layer._tpu_fused_rs_physical_order = torch.tensor(
                [0], dtype=torch.int64
            )
            prepared_layer._tpu_fused_rs_group_offset = torch.tensor(
                [0, 0], dtype=torch.int32
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
        self.assertNotIn("collective_id", call)
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
        from sglang.srt.layers.moe.pallas import fused_moe_rs_op

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
        # The shared backend owns collective-ID reuse; the adapter only issues
        # the same deterministic chunk sequence on every rank.
        self.assertTrue(
            all(
                "collective_id" not in call.kwargs
                for call in fused_moe_rs_gather.call_args_list
            )
        )
        torch.testing.assert_close(result.hidden_states, hidden_states)

    def test_fused_backend_caches_declared_post_expert_norm_weight(self):
        from sglang.srt.hardware_backend.tpu import fp4_moe

        layer = _make_block512_layer(include_unused_params=False)
        layer.hidden_size = HIDDEN_SIZE
        norm_weight = torch.ones(HIDDEN_SIZE, dtype=torch.bfloat16)
        layer.supports_tpu_fused_rs_post_expert_norm = True
        layer.tpu_fused_rs_post_expert_norm_weight = lambda: norm_weight
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

    def test_fused_backend_rejects_missing_post_expert_norm_weight_hook(self):
        from sglang.srt.hardware_backend.tpu import fp4_moe

        layer = _make_block512_layer(include_unused_params=False)
        layer.hidden_size = HIDDEN_SIZE
        # Capability declared but the weight hook is missing.
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

    def test_fused_rs_group_offset_stays_model_sharded(self):
        package_root = pathlib.Path(__file__).resolve().parents[5] / "python"
        wrapper = (
            package_root / "sglang/srt/layers/moe/pallas/fused_moe_rs_op.py"
        ).read_text()
        body = (
            package_root / "sglang/srt/layers/moe/pallas/fused_rs/fused_moe.py"
        ).read_text()

        self.assertIn('P("model"),\n            None,', wrapper)
        self.assertIn("ep_p_spec,  # group_offset", body)

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
        from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform
        from sglang.srt.layers import quantization
        from sglang.srt.layers.quantization import modelopt_quant

        platform = TpuSRTPlatform()
        self.assertIs(
            platform.get_quantization_config("modelopt_fp4"),
            modelopt_quant.TpuModelOptCompatibleFp4Config,
        )
        mixed_config_cls = platform.get_quantization_config("modelopt_mixed")
        self.assertIs(mixed_config_cls, modelopt_quant.TpuModelOptMixedPrecisionConfig)
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
                modelopt_quant.TpuModelOptCompatibleFp4Config,
            )
            self.assertIs(
                quantization.get_quantization_config("modelopt_mixed"),
                modelopt_quant.TpuModelOptMixedPrecisionConfig,
            )
        self.assertEqual(is_cpu.call_count, 2)

    def test_hybrid_loader_preserves_metadata_and_uses_platform_config(self):
        from sglang.srt.layers.quantization import modelopt_quant
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.model_loader import loader

        fp8_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            packed_modules_mapping={"gate_up_proj": ["gate_proj", "up_proj"]},
        )
        model_config = types.SimpleNamespace(
            quantization="fp8",
            is_fp4_experts=False,
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
                return_value=modelopt_quant.TpuModelOptCompatibleFp4Config,
            ) as get_quantization_config,
        ):
            config = loader._get_quantization_config(
                model_config, types.SimpleNamespace()
            )

        self.assertIsInstance(config, modelopt_quant.HybridFp8NvFp4Config)
        self.assertIsInstance(
            config.nvfp4_config, modelopt_quant.TpuModelOptCompatibleFp4Config
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
                modelopt_quant, "is_blackwell_supported", return_value=True
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
                        f"block-512 extension.*{rejected_field}",
                    ):
                        method_cls(config)

    def test_tpu_config_selects_moe_adapter_and_rejects_dense_fp4(self):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization import modelopt_quant
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        config = modelopt_quant.TpuModelOptCompatibleFp4Config.from_config(
            self._checkpoint_config()
        )
        method = config.get_quant_method(
            mock.Mock(spec=FusedMoE), "model.layers.0.mlp.experts"
        )

        self.assertIsInstance(method, modelopt_quant.TpuModelOptNvFp4FusedMoEMethod)
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

        tpu_config = modelopt_quant.TpuModelOptCompatibleFp4Config.from_config(
            self._checkpoint_config()
        )
        hybrid_config = modelopt_quant.HybridFp8NvFp4Config(
            fp8_config=types.SimpleNamespace(is_fp4_experts=False),
            nvfp4_config=tpu_config,
        )

        method = hybrid_config.get_quant_method(
            mock.Mock(spec=FusedMoE), "model.layers.0.mlp.experts"
        )
        self.assertIsInstance(method, modelopt_quant.TpuModelOptNvFp4FusedMoEMethod)

        unsupported_tpu_config = (
            modelopt_quant.TpuModelOptCompatibleFp4Config.from_config(
                self._checkpoint_config(group_size=16)
            )
        )
        unsupported_hybrid = modelopt_quant.HybridFp8NvFp4Config(
            fp8_config=types.SimpleNamespace(is_fp4_experts=False),
            nvfp4_config=unsupported_tpu_config,
        )
        with self.assertRaisesRegex(NotImplementedError, "block-512"):
            unsupported_hybrid.get_quant_method(
                mock.Mock(spec=FusedMoE), "model.layers.0.mlp.experts"
            )

    def test_tpu_config_parses_flat_checkpoint_metadata(self):
        from sglang.srt.layers.quantization import modelopt_quant

        config = modelopt_quant.TpuModelOptCompatibleFp4Config.from_config(
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
        method = modelopt_quant.TpuModelOptNvFp4FusedMoEMethod(config)
        self.assertEqual(method._weight_scale_dtype(), torch.bfloat16)
        self.assertEqual(method._prepared_scale_dtype, torch.bfloat16)

    def test_tpu_moe_accepts_and_honors_supported_scale_matrix(self):
        from sglang.srt.layers.quantization import modelopt_quant

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
                config = modelopt_quant.TpuModelOptCompatibleFp4Config.from_config(
                    self._checkpoint_config(**overrides)
                )
                method = modelopt_quant.TpuModelOptNvFp4FusedMoEMethod(config)
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
        from sglang.srt.layers.quantization import modelopt_quant

        cases = (
            ({"global_scale": "false"}, "global_scale to be a bool"),
            ({"global_scale": 1}, "global_scale to be a bool"),
            ({"global_scale": None}, "global_scale to be a bool"),
            ({"scale_dtype": 32}, "scale_dtype to be a string"),
            ({"scale_dtype": ""}, "Unsupported.*scale_dtype"),
            ({"scale_dtype": "float16"}, "Unsupported.*scale_dtype"),
        )
        for overrides, error in cases:
            config = modelopt_quant.TpuModelOptCompatibleFp4Config.from_config(
                self._checkpoint_config(**overrides)
            )
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, error):
                    modelopt_quant.TpuModelOptNvFp4FusedMoEMethod(config)

    def test_tpu_moe_rejects_non_block512_group_sizes(self):
        from sglang.srt.layers.quantization import modelopt_quant

        for group_size in (16, 256, None):
            config = modelopt_quant.TpuModelOptCompatibleFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                group_size=group_size,
                exclude_modules=[],
            )
            with self.subTest(group_size=group_size):
                with self.assertRaisesRegex(NotImplementedError, "block-512"):
                    modelopt_quant.TpuModelOptNvFp4FusedMoEMethod(config)

    def test_folded_scale_checkpoint_skips_unused_auxiliary_parameters(self):
        from sglang.srt.layers.quantization import modelopt_quant

        config = modelopt_quant.TpuModelOptCompatibleFp4Config.from_config(
            self._checkpoint_config(global_scale=False, scale_dtype="fp32")
        )
        self.assertIs(config.global_scale, False)
        self.assertEqual(config.scale_dtype, "fp32")
        layer = torch.nn.Module()
        layer.num_local_experts = EXPERTS
        layer.num_experts = EXPERTS
        layer.moe_runner_config = types.SimpleNamespace(is_gated=True)

        method = modelopt_quant.TpuModelOptNvFp4FusedMoEMethod(config)
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
    ):
        from sglang.srt.layers.moe import MoeRunnerBackend
        from sglang.srt.layers.quantization import modelopt_quant

        config = modelopt_quant.TpuModelOptCompatibleFp4Config.from_config(
            {
                "quant_method": "modelopt",
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": BLOCK_SIZE,
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
                modelopt_quant,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend(backend),
            ),
            mock.patch(
                "sglang.srt.distributed.get_world_group",
                return_value=types.SimpleNamespace(world_size=world_size),
            ),
            mock.patch(
                "sglang.srt.distributed.get_tp_group",
                return_value=types.SimpleNamespace(world_size=world_size),
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=types.SimpleNamespace(
                    attn_dp_size=1, enable_dp_attention=False
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
                "sglang.srt.hardware_backend.tpu.fp4_moe.prepare_tpu_nvfp4_moe_weights",
                _fake_prepare,
            ),
        ):
            method = modelopt_quant.TpuModelOptNvFp4FusedMoEMethod(config)
            method.create_moe_runner(layer, moe_runner_config)
            constructed = method.handles_moe_reduction
            if not model_contract:
                layer.allow_backend_reduction = True
            method.process_weights_after_loading(layer)
        return constructed, method.handles_moe_reduction, prepared, layer

    def test_gate_sees_overrides_applied_after_construction(self):
        constructed, resolved, prepared, _ = self._run_gate()

        self.assertFalse(constructed, "gate must not be decided in the constructor")
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

    def test_gate_still_declines_an_unsupported_layout(self):
        # EP 2 inside a world of 4 is the mixed TP/EP shape the requirement
        # list rejects; deferring the decision must not soften the gate.
        _, resolved, prepared, _ = self._run_gate(ep_size=2, world_size=4)

        self.assertFalse(resolved)
        self.assertFalse(prepared["use_fused_rs"])

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
