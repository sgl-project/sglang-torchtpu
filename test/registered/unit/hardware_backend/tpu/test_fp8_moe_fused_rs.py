# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TPU FP8 fused-RS runtime selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.fp8_moe_fused_rs import (
    apply_fp8_fused_rs,
    fp8_fused_rs_unsupported_reason,
    supports_fp8_fused_rs,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.pallas.fused_moe_rs_op import (
    fused_moe_rs_padded_tokens,
)
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

    def test_large_prefill_stays_on_fused_rs(self):
        tokens, hidden = 8193, 4
        x = torch.zeros(tokens, hidden)
        dispatch_output = SimpleNamespace(
            hidden_states=x,
            topk_output=(
                torch.zeros(tokens, 8),
                torch.zeros(tokens, 8, dtype=torch.int32),
                None,
            ),
        )
        layer = SimpleNamespace(
            w13_weight=torch.empty(0),
            w2_weight=torch.empty(0),
            w13_weight_scale=torch.empty(0),
            w2_weight_scale=torch.empty(0),
            _tpu_fused_rs_group_offset=torch.empty(0),
            _tpu_fused_rs_physical_order=torch.arange(8),
        )
        padded_tokens = fused_moe_rs_padded_tokens(tokens, topk=8, ep_size=8)
        self.assertGreater(padded_tokens * 8, 65536)

        with patch(
            "sglang.srt.layers.moe.pallas.fused_moe_rs_op.fused_moe_rs_gather",
            side_effect=lambda **kwargs: kwargs["hidden_states"] + 1,
        ) as fused_moe_rs:
            actual = apply_fp8_fused_rs(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
            )

        torch.testing.assert_close(actual.hidden_states, torch.ones_like(x))
        fused_moe_rs.assert_called_once()

    def test_small_route_uses_one_fused_rs_launch(self):
        tokens, hidden = 256, 4
        x = torch.zeros(tokens, hidden)
        dispatch_output = SimpleNamespace(
            hidden_states=x,
            topk_output=(
                torch.zeros(tokens, 8),
                torch.zeros(tokens, 8, dtype=torch.int32),
                None,
            ),
        )
        layer = SimpleNamespace(
            w13_weight=torch.empty(0),
            w2_weight=torch.empty(0),
            w13_weight_scale=torch.empty(0),
            w2_weight_scale=torch.empty(0),
            _tpu_fused_rs_group_offset=torch.empty(0),
            _tpu_fused_rs_physical_order=torch.arange(8),
        )

        with patch(
            "sglang.srt.layers.moe.pallas.fused_moe_rs_op.fused_moe_rs_gather",
            side_effect=lambda **kwargs: kwargs["hidden_states"] + 1,
        ) as fused_moe_rs:
            actual = apply_fp8_fused_rs(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
            )

        torch.testing.assert_close(actual.hidden_states, torch.ones_like(x))
        fused_moe_rs.assert_called_once()

    def test_padded_routed_rows_include_topk(self):
        padded_tokens = fused_moe_rs_padded_tokens(682, topk=3, ep_size=8)
        self.assertEqual(padded_tokens, 688)

    def test_skips_only_ep_reduction_for_reduced_expert_output(self):
        # Upstream's dwdp gate reads the published parallel context, which
        # unit tests don't have; pin dwdp_size=1 (the no-dwdp default). The
        # other checks on this path are context-free on TPU.
        with patch(
            "sglang.srt.layers.moe.utils.get_parallel",
            return_value=SimpleNamespace(dwdp_size=1),
        ):
            self.assertTrue(
                should_skip_post_experts_all_reduce(
                    is_tp_path=False, experts_reduced=True
                )
            )
            self.assertFalse(
                should_skip_post_experts_all_reduce(
                    is_tp_path=True, experts_reduced=True
                )
            )
            self.assertFalse(
                should_skip_post_experts_all_reduce(
                    is_tp_path=False, experts_reduced=False
                )
            )

    def test_fused_moe_exposes_quant_method_reduction_contract(self):
        layer = object.__new__(FusedMoE)
        layer.quant_method = SimpleNamespace(handles_moe_reduction=True)
        self.assertTrue(layer.handles_moe_reduction)
        layer.quant_method.handles_moe_reduction = False
        self.assertFalse(layer.handles_moe_reduction)


if __name__ == "__main__":
    unittest.main()
