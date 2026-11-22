# SPDX-License-Identifier: Apache-2.0
import sys
import unittest
from types import SimpleNamespace as N
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import fused_rs_sequence_parallel as seq
from sglang_tpu.models.muse_spark import MuseSparkForConditionalGeneration, MuseSparkMoE

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Linear(nn.Module):
    def __init__(self, a, b, dt):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((b, a), dtype=dt, device="meta"), requires_grad=False
        )

    def forward(self, x):
        return x.new_empty((x.shape[0], self.weight.shape[0])), None


class Experts(nn.Module):
    def forward(self, x, route):
        assert route == "routed"
        return x


class TestNativeEp8FromLogits(CustomTestCase):
    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires native FP4 tensor dtype"
    )
    def test_native_ep8_fused_decode_dispatch(self):
        m = MuseSparkMoE.__new__(MuseSparkMoE)
        nn.Module.__init__(m)
        m.tp_size = 8
        m._fused_decode_op = None
        m._decode_fused_router = True
        m.oneshot_gather = False
        m._manual_post_gather = True
        m.num_experts = 256
        m.top_k = 8
        m.expert_hidden_size = 4096
        m.router_multiplier = 0.17677669529663687
        m.route_scale = 1.0
        m.route_eps = 1e-15
        m.quantized_experts = True
        m.gate = Linear(8192, 256, torch.float32)
        m.pre_expert_proj = Linear(8192, 4096, torch.bfloat16)
        m.pre_expert_norm = nn.Identity()
        m.post_expert_proj = Linear(4096, 1024, torch.bfloat16)
        ex = m.experts = Experts()
        ex.handles_moe_reduction = True
        m.e_score_correction_bias = nn.Parameter(
            torch.empty((256,), device="meta"), requires_grad=False
        )
        for n, shape, dt in [
            ("w13_weight", (32, 4096, 4096), torch.float4_e2m1fn_x2),
            ("w2_weight", (32, 4096, 2048), torch.float4_e2m1fn_x2),
            ("w13_weight_scale", (32, 256, 8192), torch.bfloat16),
            ("w2_weight_scale", (32, 256, 4096), torch.bfloat16),
        ]:
            setattr(
                ex,
                n,
                nn.Parameter(
                    torch.empty(shape, device="meta", dtype=dt), requires_grad=False
                ),
            )
        ex._tpu_fused_rs_group_offset = torch.tensor([160], dtype=torch.int32)
        norm = torch.ones(4096, dtype=torch.bfloat16, device="meta")
        ex._tpu_fused_rs_has_post_expert_norm = True
        ex._tpu_fused_rs_post_expert_norm_weight = norm
        p = N(
            moe_ep_size=8,
            moe_ep_rank=5,
            tp_rank=5,
            moe_tp_size=1,
            pp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            attn_dcp_size=1,
        )
        calls = []

        def fused(rank, logits, bias, x, w1, s1, w2, s2, weight):
            assert (
                int(rank[0]) == 5
                and logits.dtype == torch.float32
                and w1 is ex.w13_weight
                and s1 is ex.w13_weight_scale
                and w2 is ex.w2_weight
                and s2 is ex.w2_weight_scale
                and weight._base is norm
            )
            calls.append("fused")
            return x

        def route(r):
            calls.append("source")
            return "routed"

        m._route = route

        def gather(x, transpose_2d=True):
            calls.append("transpose" if transpose_2d else "raw")
            return x.new_empty(
                (x.shape[0], 8192) if transpose_2d else (x.shape[0] * 8, 1024)
            )

        m._gather_columns = gather
        build = Mock(return_value=fused)
        bridge = "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op"
        previous = sys.modules.get(bridge)
        sys.modules[bridge] = N(get_fused_decode_op=build)
        if previous is None:
            self.addCleanup(sys.modules.pop, bridge, None)
        else:
            self.addCleanup(sys.modules.__setitem__, bridge, previous)
        graph = N(tpu_compile_tracer="dynamo", use_spmd_compile=False)
        with (
            patch.dict("os.environ", WORLD_SIZE="8"),
            patch("sglang_tpu.models.muse_spark.get_parallel", return_value=p),
            patch("sglang.srt.runtime_context.get_exec", return_value=N(graph=graph)),
            patch("torch.distributed.is_initialized", return_value=True) as initialized,
            patch("torch.distributed.get_rank", return_value=5) as global_rank,
        ):
            for no in (
                "tp",
                "ep",
                "rank",
                "replica",
                "stacked",
                "qblock",
                "scale",
                "router",
                "direct",
                "spmd",
                "global",
                "not_initialized",
                "fp32_norm",
            ):
                if no == "direct":
                    graph.tpu_compile_tracer = "direct"
                if no == "spmd":
                    graph.use_spmd_compile = True
                if no == "global":
                    global_rank.return_value = 7
                if no == "not_initialized":
                    initialized.return_value = False
                if no == "fp32_norm":
                    ex._tpu_fused_rs_post_expert_norm_weight = norm.float()
                if no == "tp":
                    m.tp_size = 4
                if no == "ep":
                    p.moe_ep_size = 4
                if no == "rank":
                    p.moe_ep_rank = 7
                if no == "replica":
                    m._eplb_num_replicas = 3
                if no == "stacked":
                    ex.register_buffer("_stacked_w13_weight", torch.empty(0))
                if no == "router":
                    m._decode_fused_router = False
                if no == "qblock":
                    ex._tpu_nvfp4_qblock = 256
                if no == "scale":
                    old = ex.w13_weight_scale
                    ex.w13_weight_scale = nn.Parameter(
                        old.detach()[:, :8, :], requires_grad=False
                    )
                m.prepare_fused_decode(target_ep=8)
                assert not build.called and m._fused_decode_op is None, no
                m.tp_size = 8
                p.moe_ep_size = 8
                global_rank.return_value, initialized.return_value = 5, True
                graph.tpu_compile_tracer, graph.use_spmd_compile = "dynamo", False
                ex._tpu_fused_rs_post_expert_norm_weight = norm
                p.moe_ep_rank = 5
                m._decode_fused_router = True
                m._eplb_num_replicas = 0
                ex._tpu_nvfp4_qblock = 0
                ex._buffers.pop("_stacked_w13_weight", None)
                if no == "scale":
                    ex.w13_weight_scale = old
            m.prepare_fused_decode(target_ep=8)
            assert build.call_count == 1 and set(m._buffers) == {
                "_fused_decode_rank",
                "_fused_decode_norm",
            }
            for b, decode, plan, kind in [
                (16, True, (False,) * 3, "fused"),
                (32, True, (False,) * 3, "fused"),
                (16, False, (False,) * 3, "source"),
                (64, True, (False,) * 3, "source"),
                (16, True, (False, True, False), "source"),
            ]:
                with patch.object(
                    seq, "get_fused_moe_sequence_parallel_plan", return_value=plan
                ):
                    x = torch.empty((b, 8192), dtype=torch.bfloat16, device="meta")
                    calls.clear()
                    r = m(x, x.float(), return_untransposed_gather=True, decode=decode)
                    assert calls == [kind, "raw"] and r.shape == (b * 8, 1024), (
                        b,
                        calls,
                        r.shape,
                    )

    def test_postload_only_enables_requested_native_k16(self):
        prepare = Mock()
        root = N(
            _megakernel_enabled=False,
            start_layer=0,
            end_layer=1,
            model=N(layers=[N(mlp=N(prepare_fused_decode=prepare))]),
            _compact_block256_expert_scales=Mock(),
            _probe_block256_expert_scales=Mock(),
        )
        for native, compact, probe, repeat, expected in [
            (False, False, False, False, False),
            (True, False, False, False, True),
            (True, True, False, False, False),
            (True, False, True, False, False),
            (True, False, False, True, False),
        ]:
            values = {
                "KERNEL_ENABLED": True,
                "KERNEL_DECODE_FUSED_EP16": False,
                "KERNEL_DECODE_FUSED_NATIVE_EP8": native,
                "KERNEL_COMPACT_K256_SCALES": compact,
                "PROBE_BLOCK256_SCALES": probe,
                "KERNEL_BLOCK256_SCALES": repeat,
            }
            env = N(
                **{
                    "SGLANG_TPU_MUSE_" + name: N(get=lambda v=value: v)
                    for name, value in values.items()
                }
            )
            prepare.reset_mock()
            with (
                patch("sglang_tpu.models.muse_spark.tpu_envs", env),
                patch(
                    "sglang_tpu.models.muse_spark.get_disagg",
                    return_value=N(disaggregation_mode="decode"),
                ),
            ):
                MuseSparkForConditionalGeneration.precompile_kernels_after_loading(root)
            assert prepare.call_count == int(expected)


if __name__ == "__main__":
    unittest.main()
