"""CPU checks for the opt-in EP16 compact-MoE entry, owner mapping, and routing."""

import importlib.util
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.muse_spark import MuseSparkMoE

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMuseEp16FromLogits(CustomTestCase):
    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires packed FP4 dtype"
    )
    def test_model_entry_and_fallback(self):
        moe = MuseSparkMoE.__new__(MuseSparkMoE)
        torch.nn.Module.__init__(moe)
        moe.tp_size = 16
        moe._fused_decode_op = None
        moe._decode_fused_router = True
        moe.quantized_experts = True
        moe.num_experts, moe.top_k, moe.expert_hidden_size = 256, 8, 4096
        moe.route_scale, moe.route_eps, moe.router_multiplier = (
            1.0,
            1e-15,
            0.17677669529663687,
        )
        moe.oneshot_gather = False
        moe._manual_post_gather = True
        moe.gate = Mock(
            weight=torch.empty((256, 8192), device="meta"),
            side_effect=lambda r: (r[:, :256], None),
        )
        moe.e_score_correction_bias = torch.empty(256, device="meta")
        moe.pre_expert_proj = lambda x: (x[:, :4096], None)
        moe.pre_expert_norm = torch.nn.Identity()
        moe.post_expert_proj = lambda x: (x[:, :512], None)
        moe._route = Mock(return_value=object())
        moe._gather_columns = Mock(
            side_effect=lambda x, transpose_2d: x.new_empty((16 * x.shape[0], 512))
        )
        ex = Mock(handles_moe_reduction=True, side_effect=lambda x, topk: x)
        ex._buffers = {}
        ex._tpu_fused_rs_group_offset = torch.tensor([9 * 16], dtype=torch.int32)
        ex._tpu_fused_rs_has_post_expert_norm = True
        norm = ex._tpu_fused_rs_post_expert_norm_weight = torch.empty(
            4096, device="meta", dtype=torch.bfloat16
        )
        fp4 = torch.float4_e2m1fn_x2
        for name, shape, dt in (
            ("w13_weight", (16, 4096, 4096), fp4),
            ("w2_weight", (16, 4096, 2048), fp4),
            ("w13_weight_scale", (16, 16, 8192), torch.bfloat16),
            ("w2_weight_scale", (16, 16, 4096), torch.bfloat16),
        ):
            setattr(ex, name, torch.empty(shape, dtype=dt, device="meta"))
        moe.experts = ex
        parallel = SimpleNamespace(
            moe_ep_size=16,
            moe_ep_rank=9,
            tp_rank=9,
            moe_tp_size=1,
            pp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            attn_dcp_size=1,
        )
        graph = SimpleNamespace(tpu_compile_tracer="dynamo", use_spmd_compile=False)
        fused = Mock(side_effect=lambda rank, logits, bias, x, *args: x)
        build = Mock(return_value=fused)
        bridge = "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op"
        with (
            patch.dict("os.environ", WORLD_SIZE="16"),
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(graph=graph),
            ),
            patch("sglang_tpu.models.muse_spark.get_parallel", return_value=parallel),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=9) as world_rank,
            patch.dict(
                sys.modules, {bridge: SimpleNamespace(get_fused_decode_op=build)}
            ),
        ):
            for obj, key, value in (
                (parallel, "moe_ep_rank", 7),
                (parallel, "moe_ep_size", 8),
                (graph, "use_spmd_compile", True),
                (graph, "tpu_compile_tracer", "direct"),
                (moe, "_eplb_num_replicas", 2),
                (ex, "_tpu_nvfp4_qblock", 256),
                (ex, "w13_weight_scale", ex.w13_weight_scale[:, :8]),
            ):
                with self.subTest(key=key), patch.object(obj, key, value, create=True):
                    moe.prepare_fused_decode()
                    build.assert_not_called()
            world_rank.return_value = 8
            moe.prepare_fused_decode()
            build.assert_not_called()
            world_rank.return_value = 9
            moe.prepare_fused_decode()
        build.assert_called_once_with()
        self.assertIs(moe._fused_decode_norm, norm)
        self.assertEqual(int(moe._fused_decode_rank.item()), 9)
        for b, decode, sp, want in [
            (16, True, False, True),
            (32, True, False, True),
            (64, True, False, False),
            (16, False, False, False),
            (16, True, True, False),
        ]:
            with (
                self.subTest(batch=b, decode=decode, sp=sp),
                patch(
                    "sglang_tpu.fused_rs_sequence_parallel.get_fused_moe_sequence_parallel_plan",
                    return_value=(False, sp, False),
                ),
            ):
                fused.reset_mock()
                moe._route.reset_mock()
                x = torch.empty(b, 8192, device="meta", dtype=torch.bfloat16)
                y = moe(x, x.float(), return_untransposed_gather=True, decode=decode)
                self.assertEqual(tuple(y.shape), (b * 16, 512))
                self.assertEqual(fused.called, want)
                self.assertEqual(moe._route.called, not want)
                self.assertFalse(moe._gather_columns.call_args.kwargs["transpose_2d"])
                if want:
                    self.assertIs(fused.call_args.args[4], ex.w13_weight)
                    self.assertIs(fused.call_args.args[6], ex.w2_weight)

    @unittest.skipUnless(
        importlib.util.find_spec("torch_tpu") and importlib.util.find_spec("jax"),
        "requires TPU JAX kernel library",
    )
    def test_bias_selects_unbiased_weights_ties_and_inactive_rows(self):
        import jax.numpy as jnp
        import numpy as np

        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_experts import (
            route_weights,
        )

        logits = np.zeros((16, 256), dtype=np.float32)
        bias = np.zeros((1, 256), np.float32)
        # A large correction wins a slot despite this expert's low unbiased probability.
        logits[0, 90] = -2
        bias[0, 90] = 1
        logits[1, :] = -np.inf
        result = np.asarray(route_weights(jnp.asarray(logits), jnp.asarray(bias)))
        self.assertEqual(np.flatnonzero(result[0]).tolist(), [*range(7), 90])
        self.assertLess(result[0, 90], result[0, 0])
        np.testing.assert_allclose(result[0].sum(), 1, rtol=1e-6)
        np.testing.assert_array_equal(result[1], np.zeros(256))
        np.testing.assert_array_equal(np.flatnonzero(result[2]), [*range(7), 90])


if __name__ == "__main__":
    unittest.main()
