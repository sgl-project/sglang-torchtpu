# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TPU FP8 fused-RS runtime selection."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.hardware_backend.tpu.fp8_moe_fused_rs import (
    apply_fp8_fused_rs,
    fp8_fused_rs_unsupported_reason,
    should_use_fp8_fused_rs,
    supports_fp8_fused_rs,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    should_skip_post_experts_all_reduce,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit")


class TestFp8MoeFusedRs(CustomTestCase):
    def test_exposes_first_class_moe_runner_backend(self):
        backend = MoeRunnerBackend("pallas_tpu_fused_rs")
        self.assertTrue(backend.is_pallas_tpu_fused_rs())
        self.assertFalse(backend.is_auto())

    def _config(self, **overrides):
        config = {
            "ep_size": 8,
            "tp_size": 8,
            "pp_size": 1,
            "world_size": 8,
            "reduction_compatible": True,
            "attn_dp_size": 1,
            "block_quantized": False,
            "gated": True,
            "activation": "silu",
            "apply_router_weight_on_input": False,
            "no_combine": False,
            "has_fused_shared_experts": False,
            "standard_dispatcher": True,
            "enable_dp_attention": False,
            "enable_fused_moe_sum_all_reduce": False,
        }
        config.update(overrides)
        return config

    def _supported(self, **overrides):
        return supports_fp8_fused_rs(**self._config(**overrides))

    def test_accepts_validated_ep_equals_tp_layout(self):
        self.assertTrue(self._supported())

    def test_rejects_pipeline_parallel_layout(self):
        self.assertEqual(
            fp8_fused_rs_unsupported_reason(**self._config(pp_size=2, world_size=16)),
            "pipeline parallelism is not supported, got PP=2",
        )

    def test_rejects_unvalidated_layouts_and_runner_modes(self):
        unsupported = (
            {"ep_size": 1, "tp_size": 1, "world_size": 1},
            {"ep_size": 4},
            {"world_size": 16},
            {"reduction_compatible": False},
            {"attn_dp_size": 2},
            {"block_quantized": True},
            {"gated": False},
            {"activation": "gelu"},
            {"apply_router_weight_on_input": True},
            {"no_combine": True},
            {"has_fused_shared_experts": True},
            {"standard_dispatcher": False},
            {"enable_dp_attention": True},
            {"enable_fused_moe_sum_all_reduce": True},
        )
        for overrides in unsupported:
            with self.subTest(overrides=overrides):
                self.assertFalse(self._supported(**overrides))

    def test_reports_rejected_layout(self):
        config = {
            "ep_size": 8,
            "tp_size": 8,
            "pp_size": 1,
            "world_size": 16,
            "reduction_compatible": True,
            "attn_dp_size": 1,
            "block_quantized": False,
            "gated": True,
            "activation": "silu",
            "apply_router_weight_on_input": False,
            "no_combine": False,
            "has_fused_shared_experts": False,
            "standard_dispatcher": True,
            "enable_dp_attention": False,
            "enable_fused_moe_sum_all_reduce": False,
        }
        self.assertEqual(
            fp8_fused_rs_unsupported_reason(**config),
            "expert parallel, tensor parallel, and world sizes must match, got "
            "EP=8, TP=8, world=16",
        )

    def test_large_prefill_reduces_stock_output_once(self):
        tokens, hidden = 257, 4
        x = torch.zeros(tokens, hidden)
        raw = torch.arange(tokens * hidden, dtype=torch.float32).reshape(tokens, hidden)
        dispatch_output = SimpleNamespace(
            hidden_states=x,
            topk_output=None,
        )
        layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(
                apply_router_weight_on_input=False,
                no_combine=False,
            ),
        )
        expected = raw + 1
        stock_moe = Mock(return_value=StandardCombineInput(hidden_states=raw))
        fp8_moe_module = SimpleNamespace(apply_tpu_fp8_moe=stock_moe)

        with (
            patch.dict(
                sys.modules,
                {"sglang.srt.hardware_backend.tpu.fp8_moe": fp8_moe_module},
            ),
            patch(
                "sglang.srt.distributed.moe_expert_parallel_all_reduce",
                side_effect=lambda value: value + 1,
            ) as all_reduce,
        ):
            actual = apply_fp8_fused_rs(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
            )

        torch.testing.assert_close(actual.hidden_states, expected)
        self.assertEqual(all_reduce.call_count, 1)
        self.assertFalse(stock_moe.call_args.kwargs["no_combine"])

    def test_uses_fused_rs_only_through_small_prefill_boundary(self):
        self.assertTrue(should_use_fp8_fused_rs(1))
        self.assertTrue(should_use_fp8_fused_rs(128))
        self.assertTrue(should_use_fp8_fused_rs(256))
        self.assertFalse(should_use_fp8_fused_rs(257))
        self.assertFalse(should_use_fp8_fused_rs(8192))

    def test_skips_only_ep_reduction_for_reduced_expert_output(self):
        self.assertTrue(
            should_skip_post_experts_all_reduce(is_tp_path=False, experts_reduced=True)
        )
        self.assertFalse(
            should_skip_post_experts_all_reduce(is_tp_path=True, experts_reduced=True)
        )
        self.assertFalse(
            should_skip_post_experts_all_reduce(is_tp_path=False, experts_reduced=False)
        )

    def test_fused_moe_exposes_quant_method_reduction_contract(self):
        layer = object.__new__(FusedMoE)
        layer.quant_method = SimpleNamespace(handles_moe_reduction=True)
        self.assertTrue(layer.handles_moe_reduction)
        layer.quant_method.handles_moe_reduction = False
        self.assertFalse(layer.handles_moe_reduction)


if __name__ == "__main__":
    unittest.main()
