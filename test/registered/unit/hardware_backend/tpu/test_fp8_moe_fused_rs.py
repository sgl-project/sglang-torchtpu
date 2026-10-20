# SPDX-License-Identifier: Apache-2.0

"""Unit tests for TPU FP8 fused-RS runtime selection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    post_experts_output_is_complete,
    should_skip_post_experts_all_reduce,
    sum_post_experts_output,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.fp8_moe_fused_rs import (
    _replicate_channelwise_fused_rs_scales,
    apply_fp8_fused_rs,
    fp8_fused_rs_unsupported_reason,
    supports_fp8_fused_rs,
)

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)


class TestFp8MoeFusedRs(CustomTestCase):
    def test_channelwise_scales_keep_one_layout(self):
        """Both fused-RS scales end in the same rank; the kernel rejects mixed layouts
        (expert TP16 gives w2 K=256, too small for the two-K-block form)."""
        for k2, expected in ((512, (4, 2, 64)), (256, (4, 1, 1, 64))):
            with self.subTest(k2=k2):
                layer = torch.nn.Module()
                P = lambda t: torch.nn.Parameter(t, requires_grad=False)  # noqa: E731
                layer.w13_weight = P(
                    torch.zeros(4, 1024, 64, dtype=torch.float8_e4m3fn)
                )
                layer.w2_weight = P(torch.zeros(4, k2, 64, dtype=torch.float8_e4m3fn))
                layer.w13_weight_scale = P(torch.ones(4, 1, 1, 64))
                layer.w2_weight_scale = P(torch.ones(4, 1, 1, 64))
                _replicate_channelwise_fused_rs_scales(layer)
                self.assertEqual(tuple(layer.w2_weight_scale.shape), expected)
                self.assertEqual(
                    layer.w13_weight_scale.dim(), layer.w2_weight_scale.dim()
                )

    def test_exposes_first_class_moe_runner_backend(self):
        legacy_backend = MoeRunnerBackend("pallas_tpu")
        self.assertTrue(legacy_backend.is_pallas_tpu())
        self.assertFalse(legacy_backend.is_pallas_tpu_fused_rs())

        backend = MoeRunnerBackend("pallas_tpu_fused_rs")
        self.assertTrue(backend.is_pallas_tpu_fused_rs())
        self.assertFalse(backend.is_pallas_tpu())
        self.assertFalse(backend.is_auto())

    def _config(self, **overrides):
        config = {
            "num_experts": 160,
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

    def test_accepts_block_quantized_weights(self):
        self.assertTrue(self._supported(block_quantized=True))

    def test_expert_tp_rejects_a_gate_up_width_off_256(self):
        # Qwen3-30B-A3B at TP8: I/TP = 96, so the fused gate/up width is 192.
        expert_tp = self._config(ep_size=1, expert_tp=True)
        self.assertIn(
            "multiple of 256",
            fp8_fused_rs_unsupported_reason(**expert_tp, gate_up_width=192),
        )
        self.assertIsNone(
            fp8_fused_rs_unsupported_reason(**expert_tp, gate_up_width=1024)
        )

    def test_rejects_pipeline_parallel_layout(self):
        self.assertEqual(
            fp8_fused_rs_unsupported_reason(**self._config(pp_size=2, world_size=16)),
            "pipeline parallelism is not supported, got PP=2",
        )

    def test_rejects_unvalidated_layouts_and_runner_modes(self):
        unsupported = (
            # The routing collective converts IDs to BF16, which represents
            # every supported expert index 0..255 exactly.
            {"num_experts": 257},
            {"ep_size": 1, "tp_size": 1, "world_size": 1},
            {"ep_size": 4},
            {"world_size": 16},
            {"reduction_compatible": False},
            {"attn_dp_size": 2},
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
            "num_experts": 160,
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

    def test_blockwise_path_passes_scale_inv_to_fused_kernel(self):
        tokens, hidden = 8, 4
        x = torch.zeros(tokens, hidden)
        dispatch_output = SimpleNamespace(
            hidden_states=x,
            topk_output=(
                torch.ones(tokens, 1),
                torch.zeros(tokens, 1, dtype=torch.int32),
                None,
            ),
        )
        w13_scale_inv = torch.ones(1, 1, 1, 8)
        w2_scale_inv = torch.ones(1, 1, 1, 4)
        layer = SimpleNamespace(
            weight_block_size=[1, 512],
            w13_weight=torch.zeros(1, 4, 8),
            w2_weight=torch.zeros(1, 4, 4),
            w13_weight_scale_inv=w13_scale_inv,
            w2_weight_scale_inv=w2_scale_inv,
            moe_ep_size=8,
            _tpu_fused_rs_group_offset=torch.tensor([0], dtype=torch.int32),
        )

        with patch(
            "sglang_tpu.moe.pallas.fused_moe_rs_op.fused_moe_rs_gather",
            return_value=torch.zeros_like(x),
        ) as fused:
            actual = apply_fp8_fused_rs(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
            )

        self.assertEqual(actual.hidden_states.shape, x.shape)
        self.assertIs(fused.call_args.kwargs["w1_scale"], w13_scale_inv)
        self.assertIs(fused.call_args.kwargs["w2_scale"], w2_scale_inv)
        self.assertFalse(fused.call_args.kwargs["fp8_direct_write"])

    def test_large_fused_prefill_uses_bf16_direct_write(self):
        tokens, hidden, topk = 8192, 4, 8
        x = torch.zeros(tokens, hidden)
        dispatch_output = SimpleNamespace(
            hidden_states=x,
            topk_output=(
                torch.ones(tokens, topk),
                torch.zeros(tokens, topk, dtype=torch.int32),
                None,
            ),
        )
        layer = SimpleNamespace(
            weight_block_size=None,
            w13_weight=torch.zeros(1, hidden, 2 * hidden),
            w2_weight=torch.zeros(1, hidden, hidden),
            w13_weight_scale=torch.ones(1, 1, 1, 2 * hidden),
            w2_weight_scale=torch.ones(1, 1, 1, hidden),
            moe_ep_size=8,
            _tpu_fused_rs_group_offset=torch.tensor([0], dtype=torch.int32),
        )

        with patch(
            "sglang_tpu.moe.pallas.fused_moe_rs_op.fused_moe_rs_gather",
            return_value=torch.zeros_like(x),
        ) as fused:
            apply_fp8_fused_rs(
                layer=layer,
                dispatch_output=dispatch_output,
                activation="silu",
            )

        self.assertFalse(fused.call_args.kwargs["fp8_direct_write"])

    def test_skips_only_ep_reduction_for_reduced_expert_output(self):
        # The dwdp and replicated-expert gates read the published parallel
        # context, which unit tests don't have; pin a no-dwdp, expert-parallel
        # layout. The other checks on this path are context-free on TPU.
        with patch(
            "sglang.srt.layers.moe.utils.get_parallel",
            return_value=SimpleNamespace(dwdp_size=1, tp_size=8, moe_dp_size=1),
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

    def test_boundary_leaves_the_ep_sum_to_the_fused_rs_kernel(self):
        """The fused-RS experts kernel returns the EP-summed output. The layer
        boundary owns the post-experts sum since upstream retired the model-level
        ``experts_reduced`` flag, so it has to learn from the runner backend that
        the EP reduction is done -- a boundary that sums it again doubles every
        MoE output (gsm8k 0.96 -> 0.02 on the 480B, #1122). The MoE-TP sum, when
        present, is still owed."""
        ep_calls, tp_calls = [], []
        parallel = SimpleNamespace(
            dwdp_size=1, tp_size=4, moe_ep_size=4, moe_tp_size=2, moe_dp_size=2
        )
        a2a = SimpleNamespace(
            is_flashinfer=lambda: False,
            is_pplx=lambda: False,
            is_flashinfer_megamoe=lambda: False,
        )
        with (
            patch.object(moe_utils, "_is_tpu", True),
            patch.object(moe_utils, "get_parallel", return_value=parallel),
            patch.object(moe_utils, "get_moe_a2a_backend", return_value=a2a),
            patch.object(
                moe_utils,
                "should_use_flashinfer_cutlass_moe_fp4_allgather",
                return_value=False,
            ),
            patch(
                "sglang.srt.distributed.communication_op.moe_expert_parallel_all_reduce",
                side_effect=lambda x: ep_calls.append(x) or x,
            ),
            patch(
                "sglang.srt.distributed.communication_op.moe_tensor_model_parallel_all_reduce",
                side_effect=lambda x: tp_calls.append(x) or x,
            ),
        ):
            for backend, ep_sums in (
                (MoeRunnerBackend.PALLAS_TPU_FUSED_RS, 0),
                (MoeRunnerBackend.PALLAS_TPU, 1),
            ):
                with (
                    self.subTest(backend=backend.value),
                    patch.object(
                        moe_utils, "get_moe_runner_backend", return_value=backend
                    ),
                ):
                    ep_calls.clear()
                    tp_calls.clear()
                    self.assertEqual(
                        post_experts_output_is_complete(is_tp_path=False),
                        ep_sums == 0,
                    )
                    self.assertFalse(post_experts_output_is_complete(is_tp_path=True))
                    sum_post_experts_output(torch.ones(2, 3))
                    self.assertEqual(len(ep_calls), ep_sums)
                    self.assertEqual(len(tp_calls), 1)

    def test_fused_moe_exposes_quant_method_reduction_contract(self):
        layer = object.__new__(FusedMoE)
        layer.quant_method = SimpleNamespace(handles_moe_reduction=True)
        self.assertTrue(layer.handles_moe_reduction)
        layer.quant_method.handles_moe_reduction = False
        self.assertFalse(layer.handles_moe_reduction)


if __name__ == "__main__":
    unittest.main()
