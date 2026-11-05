"""CPU coverage for Muse's unquantized TPU fused-RS adapter."""

import importlib
import os
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models import muse_spark

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def _runtime(backend="pallas_tpu_fused_rs", pp_size=1, ep_size=1):
    backend = MoeRunnerBackend(backend)
    parallel = SimpleNamespace(
        tp_size=2,
        tp_rank=0,
        tp_group=SimpleNamespace(world_size=2),
        moe_tp_size=2,
        moe_tp_rank=0,
        moe_ep_size=ep_size,
        moe_ep_rank=0,
        pp_size=pp_size,
        attn_dp_size=1,
        attn_dp_enabled=False,
    )
    with ExitStack() as stack:
        for target, value in (
            ("sglang.srt.layers.moe.get_moe_runner_backend", backend),
            ("sglang.srt.layers.quantization.unquant.get_moe_runner_backend", backend),
            ("sglang.srt.layers.moe.get_moe_a2a_backend", MoeA2ABackend.NONE),
            ("sglang.srt.runtime_context.get_parallel", parallel),
            ("sglang_tpu.models.muse_spark.get_parallel", parallel),
            (
                "sglang.srt.runtime_context.get_exec",
                SimpleNamespace(
                    moe=SimpleNamespace(enable_fused_moe_sum_all_reduce=False)
                ),
            ),
        ):
            stack.enter_context(patch(target, return_value=value))
        stack.enter_context(patch.dict(os.environ, WORLD_SIZE=str(2 * pp_size)))
        stack.enter_context(
            patch("sglang.srt.layers.quantization.unquant._is_tpu", True)
        )
        # Only TPU compilation is replaced. Layout and ownership preparation run.
        stack.enter_context(
            patch.dict(
                "sys.modules",
                {"torch_tpu._internal.pallas": SimpleNamespace(jax_op=None)},
            )
        )
        # The sys.modules patch drops modules first imported inside it, so resolve
        # the live module; Python 3.10's patch("a.b.c") can hit a stale a.b.
        for module, name in (
            ("sglang_tpu.kernels.ops.moe.fused_moe_rs_op", "_build_fused_moe_rs_op"),
            ("sglang_tpu.oneshot_collectives", "get_oneshot_all_gather_op"),
        ):
            stack.enter_context(patch.object(importlib.import_module(module), name))
        yield


def _layer():
    layer = torch.nn.Module()
    layer.num_experts = layer.num_local_experts = 2
    layer.hidden_size = 4
    layer.intermediate_size_per_partition = 128
    layer.moe_ep_size = 1
    layer.moe_tp_size = 2
    layer.num_fused_shared_experts = 0
    layer.allow_backend_reduction = True
    layer.tpu_gmm_pad = None
    layer.moe_runner_config = SimpleNamespace(
        is_gated=True,
        activation="silu",
        apply_router_weight_on_input=False,
        no_combine=True,
        routed_scaling_factor=1.0,
        gemm1_clamp_limit=None,
    )
    for name, shape in (("w13_weight", (2, 256, 4)), ("w2_weight", (2, 4, 128))):
        weight = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
        setattr(layer, name, torch.nn.Parameter(weight.bfloat16(), requires_grad=False))
    norm_weight = torch.zeros(4, dtype=torch.bfloat16)
    layer.supports_tpu_fused_rs_post_expert_norm = True
    layer.tpu_fused_rs_post_expert_norm_weight = lambda: norm_weight
    return layer


class TestBf16MoeFusedRs(CustomTestCase):
    def test_bf16_preparation_preserves_weights_and_fuses_reduction(self):
        layer = _layer()
        expected = {
            name: getattr(layer, name).detach().transpose(1, 2).contiguous()
            for name in ("w13_weight", "w2_weight")
        }
        with _runtime():
            method = muse_spark.MuseTpuBf16MoEMethod()
            method.process_weights_after_loading(layer)

        self.assertTrue(method.handles_moe_reduction)
        self.assertTrue(layer._tpu_fused_rs_expert_tp)
        self.assertEqual(layer._tpu_fused_rs_group_offset.tolist(), [0])
        for name, weight in expected.items():
            actual = getattr(layer, name)
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual, weight)
            self.assertFalse(hasattr(layer, f"{name}_scale"))

    def test_bf16_forward_passes_unscaled_weights_and_post_expert_norm(self):
        layer = _layer()
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        dispatch = SimpleNamespace(
            hidden_states=x,
            topk_output=(torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.int32), None),
        )
        with _runtime():
            method = muse_spark.MuseTpuBf16MoEMethod()
            method.moe_runner_config = layer.moe_runner_config
            method.process_weights_after_loading(layer)
            with patch(
                "sglang_tpu.kernels.ops.moe.fused_moe_rs_op.fused_moe_rs_gather",
                return_value=x * 3,
            ) as kernel:
                result = method.forward_tpu(layer, dispatch)

        torch.testing.assert_close(result.hidden_states, x * 3)
        args = kernel.call_args.kwargs
        self.assertIs(args["w1"], layer.w13_weight)
        self.assertIs(args["w2"], layer.w2_weight)
        self.assertIsNone(args["w1_scale"])
        self.assertIsNone(args["w2_scale"])
        self.assertTrue(args["has_post_expert_norm"])
        self.assertTrue(args["expert_tp"])
        self.assertFalse(args["fp8_direct_write"])

    def test_auto_keeps_standard_bf16_for_unsupported_pipeline_layout(self):
        layer = _layer()
        with _runtime("auto", pp_size=2):
            method = muse_spark.MuseTpuBf16MoEMethod()
            method.process_weights_after_loading(layer)
        self.assertFalse(method.handles_moe_reduction)
        self.assertEqual(layer.w13_weight.dtype, torch.bfloat16)

    def test_auto_keeps_standard_bf16_under_expert_parallelism(self):
        layer = _layer()
        with _runtime("auto", ep_size=2):
            method = muse_spark.MuseTpuBf16MoEMethod()
            method.process_weights_after_loading(layer)
        self.assertFalse(method.handles_moe_reduction)
        self.assertEqual(tuple(layer.w13_weight.shape), (2, 4, 256))

    def test_standard_backend_keeps_unquantized_weight_preparation(self):
        layer = _layer()
        with _runtime("pallas_tpu"):
            method = muse_spark.MuseTpuBf16MoEMethod()
            method.process_weights_after_loading(layer)
        self.assertFalse(method.handles_moe_reduction)
        self.assertEqual(tuple(layer.w13_weight.shape), (2, 4, 256))
        self.assertFalse(hasattr(layer, "_tpu_fused_rs_group_offset"))

    def test_explicit_fused_rs_rejects_unsupported_pipeline_layout(self):
        with _runtime(pp_size=2):
            method = muse_spark.MuseTpuBf16MoEMethod()
            with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
                method.process_weights_after_loading(_layer())

    def test_explicit_fused_rs_rejects_expert_parallelism(self):
        with _runtime(ep_size=2):
            method = muse_spark.MuseTpuBf16MoEMethod()
            with self.assertRaisesRegex(ValueError, "expert parallel size of 1"):
                method.process_weights_after_loading(_layer())


if __name__ == "__main__":
    unittest.main()
