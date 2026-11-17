"""DeepSeek V4's ``mlp.dp_local_forward`` seam in the non-CUDA decoder layer's
``_run_moe_ffn_dp_sync`` (``deepseek_v4_platform_stage.NonCudaDecoderMixin``)."""

import contextlib
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common import moe_transport
from sglang.srt.models.deepseek_v4_platform_stage import NonCudaDecoderMixin
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ORDINARY = ["gather", "AR", "slice"]  # DP gather, post-experts all-reduce, scatter


def _a2a_none():
    return types.SimpleNamespace(is_none=lambda: True, is_ascend_ops=lambda: False)


def run_dp_sync(dp_local_forward=None, input_ids=None, *, call=None, **kwargs):
    """Run the real method and MoE row transport on a two-rank attention-DP +
    TP-MoE fake (no weights). ``call(layer, hidden_states, batch)`` replaces the
    direct call. Returns (output, trace, batch)."""
    trace = []

    def gather(output, value, batch):
        trace.append("gather")
        output.fill_(1)

    def scatter(output, value, batch):
        trace.append("slice")
        output.copy_(value[:2])

    def mlp(value, batch, **kwargs):
        # The MoE block's own post-experts all-reduce on the gathered rows.
        trace.append("AR")
        return value * 2

    # DeepseekV2MoE.__init__ sets it to None; a backend may replace it.
    mlp.dp_local_forward = dp_local_forward
    mlp.shared_experts = None
    mlp._shared_expert_tp1 = False
    layer = types.SimpleNamespace(
        mlp=mlp, config=types.SimpleNamespace(model_type="deepseek_v41")
    )
    parallel = types.SimpleNamespace(
        attn_dp_size=2, attn_tp_size=1, tp_size=2, tp_group=None
    )
    batch = types.SimpleNamespace(
        dp_padding_mode=types.SimpleNamespace(is_max_len=lambda: True)
    )
    with contextlib.ExitStack() as stack:
        for name, value in (
            ("get_parallel", lambda: parallel),
            ("get_moe_a2a_backend", _a2a_none),
            ("is_cp_active", lambda batch: False),
            ("is_dp_gatherv_active", lambda: False),
            ("should_use_dp_reduce_scatterv", lambda: False),
            ("get_is_extend_in_batch", lambda: False),
            ("get_global_dp_buffer", lambda group: torch.empty(4, 3)),
            ("get_local_dp_buffer", lambda group: torch.empty(2, 3)),
            ("dp_gather_replicate", gather),
            ("dp_scatter", scatter),
        ):
            stack.enter_context(patch.object(moe_transport, name, value))
        stack.enter_context(patch.object(moe_transport, "SHARED_EXPERT_LOCAL", False))
        stack.enter_context(
            moe_transport.envs.SGLANG_DP_USE_REDUCE_SCATTER.override(False)
        )
        stack.enter_context(get_forward().scoped(mlp_reduce_scatter=False))
        if call is not None:
            output = call(layer, torch.ones(2, 3), batch)
        else:
            output = NonCudaDecoderMixin._run_moe_ffn_dp_sync(
                layer,
                torch.ones(2, 3),
                batch,
                input_ids=input_ids,
                input_ids_global=None,
                **kwargs,
            )
    return output, trace, batch


class TestDpLocalForward(CustomTestCase):
    def test_no_backend_keeps_the_ordinary_path(self):
        output, trace, _ = run_dp_sync()
        self.assertEqual(trace, ORDINARY)
        torch.testing.assert_close(output, torch.full((2, 3), 2.0))

    def test_backend_output_is_returned_as_is(self):
        calls = []
        local = torch.full((2, 3), 7.0)
        input_ids = torch.tensor([3, 4])

        def dp_local_forward(hidden_states, forward_batch, *, input_ids):
            calls.append((hidden_states, forward_batch, input_ids))
            return local

        output, trace, batch = run_dp_sync(dp_local_forward, input_ids)
        self.assertIs(output, local)
        self.assertEqual(trace, [], "no gather, experts all-reduce or scatter")
        ((hidden_states, forward_batch, ids),) = calls
        torch.testing.assert_close(hidden_states, torch.ones(2, 3))
        self.assertIs(forward_batch, batch)
        self.assertIs(ids, input_ids)

    def test_none_selects_the_ordinary_path(self):
        calls = []

        def decline(hidden_states, forward_batch, *, input_ids):
            calls.append(hidden_states)
            return None

        output, trace, _ = run_dp_sync(decline)
        self.assertEqual(len(calls), 1)
        self.assertEqual(trace, ORDINARY)
        torch.testing.assert_close(output, torch.full((2, 3), 2.0))

    def test_a_complete_output_cannot_be_deferred(self):
        # The MoE post may ask for a deferred MoEOutput; an output the backend
        # already reduced must not be handed to it.
        def dp_local_forward(hidden_states, forward_batch, *, input_ids):
            return torch.zeros(2, 3)

        with self.assertRaisesRegex(RuntimeError, "return_moe_output"):
            run_dp_sync(dp_local_forward, return_moe_output=True)
        # A declining backend leaves the request to the ordinary path.
        output, trace, _ = run_dp_sync(lambda *a, **k: None, return_moe_output=True)
        self.assertEqual(trace, ORDINARY)

    def test_the_platform_layers_take_the_seam(self):
        # TPU and the other non-CUDA platforms build DeepseekV4PlatformDecoderLayer;
        # DSpark draft stages reach the same method through _run_ffn.
        from sglang.srt.models.deepseek_v4 import DeepseekV4PlatformDecoderLayer
        from sglang.srt.models.deepseek_v4_dspark import DSparkV4PlatformStage
        from sglang.srt.models.deepseek_v4_platform_stage import NonCudaDSparkMixin

        for cls in (DeepseekV4PlatformDecoderLayer, DSparkV4PlatformStage):
            self.assertIs(
                cls._run_moe_ffn_dp_sync, NonCudaDecoderMixin._run_moe_ffn_dp_sync
            )

        def run_ffn(layer, hidden_states, batch):
            layer.dim = 3
            layer._run_moe_ffn_dp_sync = types.MethodType(
                NonCudaDecoderMixin._run_moe_ffn_dp_sync, layer
            )
            return NonCudaDSparkMixin._run_ffn(layer, hidden_states, batch)

        output, trace, _ = run_dp_sync(call=run_ffn)
        self.assertEqual(trace, ORDINARY)
        torch.testing.assert_close(output, torch.full((2, 3), 2.0))
        # The draft passes no token ids; a backend that took a draft layer over
        # would get input_ids=None (the TPU backend never prepares is_nextn
        # layers, which DSpark stages are).
        seen = []
        run_dp_sync(
            lambda h, b, *, input_ids: seen.append(input_ids) or h, call=run_ffn
        )
        self.assertEqual(seen, [None])


if __name__ == "__main__":
    unittest.main()
