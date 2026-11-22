# SPDX-License-Identifier: Apache-2.0
"""CPU unit + TPU7x Mosaic export tests for Wave-5 DFlash d=6 (B=384) fused decode."""

import functools
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from sglang.srt.configs.muse_spark import MuseSparkTextConfig
from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.decode_qkv_epilogue_tpu import can_use_muse_decode_qkv_epilogue
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.fp8_weight_linear import TpuFp8LinearMethod, _fp8_matmul_jax
from sglang_tpu.models.muse_spark import (
    MuseSparkDecoderLayer,
    MuseSparkGatedResidual,
    MuseSparkMoE,
    MuseSparkRMSNorm,
    _decode_max_tokens,
)
from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
    SUPPORTED_FUSE_ALL_BATCHES,
    SUPPORTED_FUSE_ALL_BATCHES_B384,
    _validate_fuse_all_shapes,
    supported_fuse_all_batches,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_stub_moe(*, world: int = 8, ep_size: int = 8, tp_experts: int = 1, block: int = 256):
    moe = MuseSparkMoE.__new__(MuseSparkMoE)
    nn.Module.__init__(moe)
    moe.tp_size = world
    moe.moe_ep_size = ep_size
    moe.moe_tp_size = tp_experts
    moe.num_experts = 256
    moe.top_k = 8
    moe.expert_hidden_size = 4096
    moe.route_scale = 1.0
    moe.route_eps = 1e-15
    moe.router_multiplier = 0.17677669529663687
    moe._is_hybrid_ep_tp2 = tp_experts == 2
    moe._fused_decode_op = None
    moe._fused_decode_batches = (16, 32)
    moe._fused_decode_v2_op = None
    moe._fused_decode_v2_batches = ()
    moe._fused_decode_num_replicas = 0
    moe._fused_decode_mode = ""
    moe._decode_allow_prefill_eplb = False
    moe._decode_sparse_v2 = False
    moe._decode_sparse_fp8 = False
    moe._decode_fused_router = 1
    moe._manual_post_gather = True
    moe.oneshot_gather = False
    moe.quantized_experts = True

    moe.gate = nn.Module()
    moe.gate.weight = nn.Parameter(
        torch.zeros((256, 8192), dtype=torch.float32), requires_grad=False
    )
    moe.e_score_correction_bias = nn.Parameter(
        torch.zeros((256,), dtype=torch.float32), requires_grad=False
    )
    moe.pre_expert_proj = nn.Module()
    moe.pre_expert_proj.weight = nn.Parameter(
        torch.zeros((4096 // world, 8192), dtype=torch.bfloat16), requires_grad=False
    )
    moe.pre_expert_norm = MuseSparkRMSNorm(4096, eps=1e-5, gain_center=1.0)
    moe.pre_expert_norm.weight.data.zero_()
    moe.pre_expert_norm.precompute_gain()
    moe.post_expert_norm = MuseSparkRMSNorm(
        4096, eps=1e-8, gain_center=1.0, scale_input=True
    )
    moe.post_expert_norm.weight.data.zero_()
    moe.post_expert_norm.precompute_gain()
    moe.post_expert_proj = nn.Module()
    moe.post_expert_proj.weight = nn.Parameter(
        torch.zeros((8192 // world, 4096), dtype=torch.bfloat16), requires_grad=False
    )

    e_owner = 256 // ep_size
    ex = nn.Module()
    ex.handles_moe_reduction = tp_experts == 1
    ex._tpu_fused_rs_group_offset = torch.tensor([0], dtype=torch.int32)
    ex._tpu_fused_rs_has_post_expert_norm = True
    ex._tpu_fused_rs_post_expert_norm_weight = torch.zeros(4096, dtype=torch.bfloat16)
    if tp_experts == 2:
        ex.w13_weight = nn.Parameter(
            torch.empty((e_owner, 4096, 2048), dtype=torch.float4_e2m1fn_x2, device="meta"),
            requires_grad=False,
        )
        ex.w2_weight = nn.Parameter(
            torch.empty((e_owner, 2048, 2048), dtype=torch.float4_e2m1fn_x2, device="meta"),
            requires_grad=False,
        )
        ex.w13_weight_scale = nn.Parameter(
            torch.empty((e_owner, 4096 // block, 4096), dtype=torch.bfloat16, device="meta"),
            requires_grad=False,
        )
        ex.w2_weight_scale = nn.Parameter(
            torch.empty((e_owner, 2048 // block, 4096), dtype=torch.bfloat16, device="meta"),
            requires_grad=False,
        )
    else:
        ex.w13_weight = nn.Parameter(
            torch.empty((e_owner, 4096, 4096), dtype=torch.float4_e2m1fn_x2, device="meta"),
            requires_grad=False,
        )
        ex.w2_weight = nn.Parameter(
            torch.empty((e_owner, 4096, 2048), dtype=torch.float4_e2m1fn_x2, device="meta"),
            requires_grad=False,
        )
        ex.w13_weight_scale = nn.Parameter(
            torch.empty((e_owner, 4096 // block, 8192), dtype=torch.bfloat16, device="meta"),
            requires_grad=False,
        )
        ex.w2_weight_scale = nn.Parameter(
            torch.empty((e_owner, 4096 // block, 4096), dtype=torch.bfloat16, device="meta"),
            requires_grad=False,
        )
    moe.experts = ex
    parallel = SimpleNamespace(
        tp_size=world,
        tp_rank=0,
        moe_ep_size=ep_size,
        moe_ep_rank=0,
        moe_tp_size=tp_experts,
        moe_tp_rank=0,
        moe_dp_size=1,
        pp_size=1,
        attn_dp_size=1,
        attn_cp_size=1,
        attn_dcp_size=1,
    )
    return moe, parallel


class TestMuseDecodeB384GatesAndDispatch(CustomTestCase):
    def test_default_off_and_opt_in_caps(self):
        self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.get())
        self.assertEqual(_decode_max_tokens(), 256)
        self.assertEqual(supported_fuse_all_batches(), SUPPORTED_FUSE_ALL_BATCHES)

        with tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.override(True):
            self.assertEqual(_decode_max_tokens(), 384)
            self.assertEqual(
                supported_fuse_all_batches(), SUPPORTED_FUSE_ALL_BATCHES_B384
            )

        # Verify _pad_batch_pow2 & distributed_topk_pallas B=384 gate (#811 §7 / #862 §1)
        from sglang_tpu.kernels.fp8_lm_head_fused import _pad_batch_pow2

        self.assertEqual(_pad_batch_pow2(320), 384)
        self.assertEqual(_pad_batch_pow2(384), 384)
        with self.assertRaises(ValueError):
            _pad_batch_pow2(385)
        import jax
        import jax.numpy as jnp

        _validate_fuse_all_shapes(
            "sglang_tpu_pallas::muse_decode_fp8_sparse_ep8_r0",
            "fp8_sparse",
            256,
            jax.ShapeDtypeStruct((384, 4096), jnp.bfloat16),
            jax.ShapeDtypeStruct((32, 4096, 8192), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((32, 8192), jnp.bfloat16),
        )

    def test_qkv_epilogue_b384_gate(self):
        qkv_384 = torch.zeros((384, 1280), dtype=torch.bfloat16)
        qkv_400 = torch.zeros((400, 1280), dtype=torch.bfloat16)
        rope = SimpleNamespace(cos_sin_pairs=torch.zeros((64, 128)))
        backend = SimpleNamespace(can_forward_decode_prepared=lambda *_: True)
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            attn_backend=backend,
        )
        with tpu_envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.override(True):
            self.assertFalse(
                can_use_muse_decode_qkv_epilogue(qkv_384, rope, object(), fb)
            )
            with tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.override(True):
                self.assertTrue(
                    can_use_muse_decode_qkv_epilogue(qkv_384, rope, object(), fb)
                )
                self.assertFalse(
                    can_use_muse_decode_qkv_epilogue(qkv_400, rope, object(), fb)
                )

        # Verify decode_attn_gate_fuse is restricted to DECODE (dq=1) and bypassed on
        # TARGET_VERIFY (dq>1) so split views don't perturb spec verification (#896 §2).
        from sglang_tpu.models.muse_spark import MuseSparkAttention

        attn = MuseSparkAttention.__new__(MuseSparkAttention)
        nn.Module.__init__(attn)
        attn.head_dim = 128
        attn.num_heads = 2
        attn.num_kv_heads = 1
        attn.q_size = 256
        attn.kv_size = 128
        attn.norm_eps = 1e-5
        attn.decode_qkv_epilogue = True
        attn.decode_attn_gate_fuse = True
        attn.query_scale = 1.0
        attn.rotary_emb = None
        attn._qkvg_w8k = None
        attn._qkvg_weight = None
        attn._o_proj_w8k = None
        attn.qkv_proj = Mock(
            side_effect=lambda x: (torch.zeros(x.shape[0], 512, dtype=x.dtype), None)
        )
        attn.gate_proj = Mock(
            side_effect=lambda x: (torch.ones(x.shape[0], 256, dtype=x.dtype), None)
        )
        attn.o_proj = Mock(side_effect=lambda x, skip_all_reduce=False: (x, None))
        attn.attn = Mock()
        attn._per_head_normalize = Mock(side_effect=lambda x: x)
        seen_gate = []

        def _fake_qkv_attn(*args, gate=None, return_post_fused=False, **kw):
            seen_gate.append((gate is not None, return_post_fused))
            out = torch.ones((args[1].shape[0], 256), dtype=torch.bfloat16)
            return (out, gate is not None) if return_post_fused else out

        with (
            patch(
                "sglang_tpu.models.muse_spark.can_use_muse_decode_qkv_epilogue",
                return_value=True,
            ),
            patch(
                "sglang_tpu.models.muse_spark.muse_decode_fused_qkv_attn",
                side_effect=_fake_qkv_attn,
            ),
        ):
            x_tv = torch.ones((384, 512), dtype=torch.bfloat16)
            pos_tv = torch.arange(384, dtype=torch.int32)
            attn(pos_tv, x_tv, SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY))
            self.assertEqual(seen_gate, [(False, False)])
            seen_gate.clear()
            attn(
                pos_tv[:64],
                x_tv[:64],
                SimpleNamespace(forward_mode=ForwardMode.DECODE),
            )
            self.assertEqual(seen_gate, [(True, True)])

    def test_dflash_fp8_linear_max_rows_gate(self):
        lin = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)
        with patch(
            "sglang_tpu.fp8_weight_linear.get_fp8_matmul_op",
            return_value=lambda x, w, s: torch.zeros(
                (x.shape[0], w.shape[1]), dtype=torch.float32
            ),
        ):
            with tpu_envs.SGLANG_TPU_DFLASH_FP8_DRAFT.override(False):
                with tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.override(True):
                    m0 = TpuFp8LinearMethod(lin)
                    self.assertEqual(m0._max_rows, 64)

            with tpu_envs.SGLANG_TPU_DFLASH_FP8_DRAFT.override(True):
                m256 = TpuFp8LinearMethod(lin)
                self.assertEqual(m256._max_rows, 256)
                with tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.override(True):
                    m384 = TpuFp8LinearMethod(lin)
                    self.assertEqual(m384._max_rows, 384)

    def test_decoder_layer_o_proj_ar_and_norm_b384_gates(self):
        hidden = 1024
        layer = MuseSparkDecoderLayer.__new__(MuseSparkDecoderLayer)
        nn.Module.__init__(layer)
        layer.input_layernorm = MuseSparkRMSNorm(hidden, eps=1e-5, gain_center=1.0)
        layer.post_attention_layernorm = MuseSparkRMSNorm(
            hidden, eps=1e-8, has_weight=False
        )
        layer.post_attention_residual_gate = MuseSparkGatedResidual(
            hidden, temperature=1.0
        )
        layer.pre_feedforward_layernorm = MuseSparkRMSNorm(
            hidden, eps=1e-5, gain_center=1.0
        )
        layer.post_feedforward_layernorm = MuseSparkRMSNorm(
            hidden, eps=1e-8, gain_center=1.0, scale_input=True
        )
        layer.post_feedforward_residual_gate = MuseSparkGatedResidual(
            hidden, temperature=1.0
        )
        for m in layer.modules():
            if isinstance(m, MuseSparkRMSNorm) and m.weight is not None:
                m.weight.data.zero_()
                m.precompute_gain()
            elif isinstance(m, MuseSparkGatedResidual):
                m.gate.data.zero_()
                m.precompute_coefficients()

        o_proj = RowParallelLinear.__new__(RowParallelLinear)
        nn.Module.__init__(o_proj)
        o_proj.tp_size = 8
        o_proj.reduce_results = True
        o_proj.bias = None
        o_proj.output_size = hidden
        o_proj.use_dp_attention_reduce = False
        o_proj.use_decode_attn_tp = False
        layer.self_attn = SimpleNamespace(o_proj=o_proj, head_dim=128)
        layer._decode_attn_qkv_tpar_auto = True

        fb = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "8"}),
            patch(
                "sglang.srt.runtime_context.get_forward",
                return_value=SimpleNamespace(sp_active=False),
            ),
        ):
            layer._decode_fused_o_proj_ar = 2
            self.assertEqual(layer._active_fused_o_proj_ar_mode(384, fb), 0)
            with tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.override(True):
                self.assertEqual(layer._active_fused_o_proj_ar_mode(384, fb), 2)
                for mode in (1, 3):
                    layer._decode_fused_o_proj_ar = mode
                    self.assertEqual(layer._active_fused_o_proj_ar_mode(384, fb), 0)

        # Verify tp_ag_res_norm (kern_med = 17.83 us at B=384 on w5p6, 0/0 spills)
        # stays gated off at B=384 by default and activates at B=384 under DECODE_B384=1 (#876 §1).
        layer._decode_norm_layout = 5
        layer._decode_fuse_target_verify = True
        layer._dummy_gain = torch.ones(hidden)
        captured_untransposed: list[bool] = []

        def _fake_mlp(h, r, *, return_untransposed_gather=False, decode=False):
            captured_untransposed.append(bool(return_untransposed_gather))
            return torch.zeros(
                (8 * h.shape[0], hidden // 8)
                if return_untransposed_gather
                else (h.shape[0], hidden),
                dtype=torch.bfloat16,
            )

        layer.mlp = Mock(
            side_effect=_fake_mlp,
            tp_size=8,
            fuses_post_gather=lambda *a, **kw: False,
        )
        layer.self_attn = Mock(side_effect=lambda p, x, fb_arg: x)
        layer.self_attn.o_proj = o_proj
        layer._decode_fused_o_proj_ar = 0
        layer._decode_attn_qkv_tpar_auto = False

        fake_decode_ops = SimpleNamespace(
            get_muse_decode_rmsnorm_op=lambda: (
                lambda x, *a: x.to(torch.bfloat16) if a[-1] else x.float()
            ),
            get_muse_decode_res_norm_op=lambda: (lambda s, *a: (s, s.to(torch.bfloat16))),
            get_muse_decode_res_only_op=lambda: (lambda s, *a: s),
            get_muse_decode_tp_ag_res_only_op=lambda: (lambda s, *a: s),
            get_muse_decode_tp_ag_res_norm_op=lambda: (
                lambda s, *a: (s, s.to(torch.bfloat16))
            ),
        )
        with patch.dict(
            sys.modules,
            {"sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router": fake_decode_ops},
        ):
            s_384 = torch.zeros((384, hidden), dtype=torch.float32)
            pos_384 = torch.zeros((384,), dtype=torch.int64)
            captured_untransposed.clear()
            layer(pos_384, s_384, fb, next_input_layernorm=layer.input_layernorm)
            self.assertEqual(captured_untransposed, [False])
            with tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.override(True):
                for bs in (256, 384):
                    captured_untransposed.clear()
                    s_in = torch.zeros((bs, hidden), dtype=torch.float32)
                    pos_in = torch.zeros((bs,), dtype=torch.int64)
                    layer(pos_in, s_in, fb, next_input_layernorm=layer.input_layernorm)
                    self.assertEqual(captured_untransposed, [True], f"bs={bs}")

        # Verify fused_fp8_lm_head_distributed_top_k B=320/384 gate (#811 §7 / #862 §1).
        from sglang_tpu.distributed_topk_pallas import (
            fused_fp8_lm_head_distributed_top_k,
        )
        from test.registered.unit.hardware_backend.tpu.test_muse_fp8_lm_head import (
            _make_fake_lm_head,
        )

        group = SimpleNamespace(world_size=8, rank_in_group=0, ranks=tuple(range(8)))
        head = _make_fake_lm_head(
            local_width=256,
            hidden_size=64,
            org_vocab_size=2048,
            global_vocab_size=2048,
            org_vocab_padded_count=256,
        )
        mock_lmh_op = Mock(
            side_effect=lambda x, *a: (
                torch.zeros(x.shape[0], a[6], dtype=torch.float32),
                torch.zeros(x.shape[0], a[6], dtype=torch.int32),
            )
        )
        with (
            patch("sglang_tpu.top_k._group_spans_world", return_value=True),
            patch(
                "sglang_tpu.kernels.fp8_lm_head_fused.get_fp8_lm_head_distributed_topk_op",
                return_value=mock_lmh_op,
            ),
            patch("torch.compiler.is_compiling", return_value=True),
        ):
            for rows, k_req in ((320, 1), (384, 1), (384, 64)):
                fx = Mock(spec=torch.Tensor)
                fx.device = SimpleNamespace(type="tpu")
                fx.ndim = 2
                fx.shape = (rows, 64)
                self.assertIsNone(
                    fused_fp8_lm_head_distributed_top_k(
                        fx, head, top_k=k_req, logit_scale=1.0, group=group
                    )
                )
                with tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.override(True):
                    res = fused_fp8_lm_head_distributed_top_k(
                        fx, head, top_k=k_req, logit_scale=1.0, group=group
                    )
                    self.assertIsNotNone(res)
                    self.assertEqual(tuple(res[1].shape), (rows, k_req))

    @unittest.skipUnless(
        hasattr(torch, "float4_e2m1fn_x2"), "requires packed FP4 dtype"
    )
    def test_moe_b384_batches_and_full_moe_hi_fallback(self):
        bridge = "sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op"
        graph = SimpleNamespace(tpu_compile_tracer="dynamo", use_spmd_compile=False)
        sparse_op = Mock(side_effect=lambda *a: a[3])
        full_op = Mock(side_effect=lambda *a: torch.zeros((a[1].shape[0], 8192)))
        fake_mod = SimpleNamespace(
            SUPPORTED_FUSE_ALL_BATCHES=(16, 32, 64, 128, 256),
            get_muse_fuse_all_decode_op=Mock(return_value=sparse_op),
            get_muse_full_moe_decode_op=Mock(return_value=full_op),
            get_muse_ep_tp2_fallback_op=Mock(return_value=Mock()),
        )

        class _ProjStub(nn.Module):
            def __init__(self, in_f: int, w_out: int, f_out: int, dtype: torch.dtype):
                super().__init__()
                self.f_out = f_out
                self.weight = nn.Parameter(
                    torch.zeros((w_out, in_f), dtype=dtype), requires_grad=False
                )

            def forward(self, x: torch.Tensor):
                return torch.zeros((x.shape[0], self.f_out), dtype=x.dtype), None

        moe, parallel = _make_stub_moe(world=8, ep_size=8, tp_experts=1, block=256)
        moe.gate = _ProjStub(8192, 256, 256, torch.float32)
        moe.pre_expert_proj = _ProjStub(8192, 512, 4096, torch.bfloat16)
        moe.post_expert_proj = _ProjStub(4096, 1024, 1024, torch.bfloat16)
        moe._gather_columns = Mock(
            side_effect=lambda s, transpose_2d=True: torch.zeros(
                (s.shape[0], 8192) if transpose_2d else (8 * s.shape[0], 1024),
                dtype=s.dtype,
            )
        )

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "8"}),
            patch(
                "sglang.srt.runtime_context.get_exec",
                return_value=SimpleNamespace(graph=graph),
            ),
            patch("sglang_tpu.models.muse_spark.get_parallel", return_value=parallel),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch(
                "sglang_tpu.fused_rs_sequence_parallel.get_fused_moe_sequence_parallel_plan",
                return_value=(False, False),
            ),
            patch.dict(sys.modules, {bridge: fake_mod}),
        ):
            moe.prepare_fused_decode(mode="sparse", num_replicas=0)
            self.assertNotIn(384, moe._fused_decode_batches)

            with tpu_envs.SGLANG_TPU_MUSE_DECODE_B384.override(True):
                moe.prepare_fused_decode(mode="full_moe", num_replicas=0)
                self.assertIn(384, moe._fused_decode_batches)
                self.assertFalse(moe.fuses_post_gather(384, decode=True))

                hs = torch.zeros((384, 8192), dtype=torch.bfloat16)
                ri = torch.zeros((384, 8192), dtype=torch.float32)
                out = moe(hs, ri, return_untransposed_gather=True, decode=True)
                self.assertEqual(tuple(out.shape), (8 * 384, 1024))
                self.assertEqual(full_op.call_count, 0)
                self.assertEqual(sparse_op.call_count, 1)

                # K16 sparse_v2 includes 384 under DECODE_B384=1 (-145 us/L over v1).
                moe_k16, par_k16 = _make_stub_moe(
                    world=8, ep_size=8, tp_experts=1, block=16
                )
                moe_k16._decode_sparse_v2 = True
                with patch(
                    "sglang_tpu.models.muse_spark.get_parallel", return_value=par_k16
                ):
                    moe_k16.prepare_fused_decode(mode="sparse", num_replicas=0)
                    self.assertEqual(
                        moe_k16._fused_decode_v2_batches, (32, 64, 128, 256, 384)
                    )

                # ep_tp2 stays capped at <= 256 even when DECODE_B384=1.
                moe_tp2, par_tp2 = _make_stub_moe(
                    world=8, ep_size=4, tp_experts=2, block=256
                )
                with patch(
                    "sglang_tpu.models.muse_spark.get_parallel", return_value=par_tp2
                ):
                    moe_tp2.prepare_fused_decode(mode="ep_tp2", num_replicas=0)
                    self.assertNotIn(384, moe_tp2._fused_decode_batches)

    def test_vpu_imbalance_loads_exact_above_256(self):
        import jax.numpy as jnp
        from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
            precompute_imbalance_loads_vpu,
        )

        # Adversarial B=384 counts: expert 0 gets all 384 tokens, owner 0 gets 3072 routed rows.
        counts = jnp.zeros((256,), dtype=jnp.int32).at[0].set(384).at[1].set(383).at[2].set(257)
        for mode in ("active", "tiles", "weighted"):
            packed = precompute_imbalance_loads_vpu(
                counts, num_owners=8, cost_mode=mode, tile_m=64
            )
            self.assertEqual(int(packed[0, 0]), 384)
            self.assertEqual(int(packed[0, 1]), 383)
            self.assertEqual(int(packed[0, 2]), 257)
            self.assertEqual(int(packed[2, 0]), 3)  # active experts on owner 0
            self.assertEqual(int(packed[3, 0]), 384 + 383 + 257)  # exact rank_rows > 256!
            if mode == "weighted":
                self.assertEqual(int(packed[1, 0]), (18 + 384) + (18 + 383) + (18 + 257))
            elif mode == "tiles":
                self.assertEqual(int(packed[1, 0]), 6 + 6 + 5)


class TestMuseDecodeB384Tpu7xMosaicExport(CustomTestCase):
    def test_tpu7x_mosaic_export_full_b384_stack(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax import export
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        from sglang_tpu import pallas_export
        from sglang_tpu.attention.pallas import util as rpa_util
        from sglang_tpu.attention.pallas.decode_qkv_epilogue import (
            build_decode_rope_step_tables,
            run_decode_qkv_epilogue,
        )
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )
        from sglang_tpu.kernels.router_topk import router_topk_jax
        from sglang_tpu.kernels.twoshot_all_reduce import bridge_twoshot_all_reduce
        from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
            bridge_all_reduce_jax,
            fp8_linear_bf16_jax,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
            residual_and_next_norm_pallas_jax,
            residual_only_pallas_jax,
            rmsnorm_pallas_jax,
            tp_ag_res_norm_pallas_jax,
            tp_ag_res_only_pallas_jax,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
            default_sparse_decode_config,
            fused_decode_sparse_fp8_from_logits,
            fused_decode_sparse_from_logits,
        )
        from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse_v2 import (
            fused_decode_sparse_v2_from_logits,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        devs = jax.devices("cpu")
        if len(devs) < 8:
            self.skipTest("Requires XLA_FLAGS=--xla_force_host_platform_device_count>=8")

        b = 384
        hidden = 8192
        mesh8 = Mesh(np.asarray(devs[:8]), ("model",))
        s_rep = NamedSharding(mesh8, P())
        s_mod = NamedSharding(mesh8, P("model"))

        with (
            patch.object(
                pallas_export, "get_torch_tpu_device_kind", return_value="TPU v7"
            ),
            patch.object(
                rpa_util, "get_torch_tpu_device_kind", return_value="TPU v7"
            ),
            tpu_info_for_cpu_export(),
        ):
            # 1. Sparse MoE B=384: K256 & K16 (v1 + v2), R=0 & R=2 (static_holder_mask)
            rank_s = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=s_rep)
            logits_s = jax.ShapeDtypeStruct((b, 256), jnp.float32, sharding=s_rep)
            bias_s = jax.ShapeDtypeStruct((1, 256), jnp.float32, sharding=s_rep)
            x_s = jax.ShapeDtypeStruct((b, 4096), jnp.bfloat16, sharding=s_rep)
            norm_s = jax.ShapeDtypeStruct((1, 4096), jnp.float32, sharding=s_rep)
            meta_r2 = jax.ShapeDtypeStruct((4, 16), jnp.int32, sharding=s_rep)

            for block in (256, 16):
                w1_s = jax.ShapeDtypeStruct((32, 4096, 8192), jnp.float4_e2m1fn, sharding=s_rep)
                s1_s = jax.ShapeDtypeStruct((32, 4096 // block, 8192), jnp.bfloat16, sharding=s_rep)
                w2_s = jax.ShapeDtypeStruct((32, 4096, 4096), jnp.float4_e2m1fn, sharding=s_rep)
                s2_s = jax.ShapeDtypeStruct((32, 4096 // block, 4096), jnp.bfloat16, sharding=s_rep)
                rw1_s = jax.ShapeDtypeStruct((2, 4096, 8192), jnp.float4_e2m1fn, sharding=s_rep)
                rs1_s = jax.ShapeDtypeStruct((2, 4096 // block, 8192), jnp.bfloat16, sharding=s_rep)
                rw2_s = jax.ShapeDtypeStruct((2, 4096, 4096), jnp.float4_e2m1fn, sharding=s_rep)
                rs2_s = jax.ShapeDtypeStruct((2, 4096 // block, 4096), jnp.bfloat16, sharding=s_rep)

                cfg_stat = dataclasses_replace_stat(
                    default_sparse_decode_config(b, 8, block)
                )
                f_r2 = jax.jit(
                    jax.shard_map(
                        functools.partial(
                            fused_decode_sparse_from_logits,
                            ep=8,
                            replica_cost_mode="tiles",
                            config=cfg_stat,
                        ),
                        mesh=mesh8,
                        in_specs=(P(),) * 14,
                        out_specs=P(),
                        check_vma=False,
                    )
                )
                exp_r2 = export.export(f_r2, platforms=["tpu"])(
                    rank_s,
                    logits_s,
                    bias_s,
                    x_s,
                    w1_s,
                    s1_s,
                    w2_s,
                    s2_s,
                    norm_s,
                    rw1_s,
                    rs1_s,
                    rw2_s,
                    rs2_s,
                    meta_r2,
                )
                self.assertIn("stablehlo.custom_call", exp_r2.mlir_module())

                if block == 16:
                    f_v2 = jax.jit(
                        jax.shard_map(
                            functools.partial(
                                fused_decode_sparse_v2_from_logits,
                                ep=8,
                                replica_cost_mode="tiles",
                            ),
                            mesh=mesh8,
                            in_specs=(P(),) * 9,
                            out_specs=P(),
                            check_vma=False,
                        )
                    )
                    exp_v2 = export.export(f_v2, platforms=["tpu"])(
                        rank_s, logits_s, bias_s, x_s, w1_s, s1_s, w2_s, s2_s, norm_s
                    )
                    self.assertIn("stablehlo.custom_call", exp_v2.mlir_module())

            # 2. L0/L61 FP8 Sparse Decode at B=384 (ring3_8m)
            fw1_s = jax.ShapeDtypeStruct((32, 4096, 8192), jnp.float8_e4m3fn, sharding=s_rep)
            fs1_s = jax.ShapeDtypeStruct((32, 8192), jnp.float32, sharding=s_rep)
            fw2_s = jax.ShapeDtypeStruct((32, 4096, 4096), jnp.float8_e4m3fn, sharding=s_rep)
            fs2_s = jax.ShapeDtypeStruct((32, 4096), jnp.float32, sharding=s_rep)
            frw1_s = jax.ShapeDtypeStruct((2, 4096, 8192), jnp.float8_e4m3fn, sharding=s_rep)
            frs1_s = jax.ShapeDtypeStruct((2, 8192), jnp.float32, sharding=s_rep)
            frw2_s = jax.ShapeDtypeStruct((2, 4096, 4096), jnp.float8_e4m3fn, sharding=s_rep)
            frs2_s = jax.ShapeDtypeStruct((2, 4096), jnp.float32, sharding=s_rep)
            f_fp8 = jax.jit(
                jax.shard_map(
                    functools.partial(
                        fused_decode_sparse_fp8_from_logits,
                        ep=8,
                        replica_cost_mode="tiles",
                    ),
                    mesh=mesh8,
                    in_specs=(P(),) * 14,
                    out_specs=P(),
                    check_vma=False,
                )
            )
            exp_fp8 = export.export(f_fp8, platforms=["tpu"])(
                rank_s,
                logits_s,
                bias_s,
                x_s,
                fw1_s,
                fs1_s,
                fw2_s,
                fs2_s,
                norm_s,
                frw1_s,
                frs1_s,
                frw2_s,
                frs2_s,
                meta_r2,
            )
            self.assertIn("stablehlo.custom_call", exp_fp8.mlir_module())

            # 3. Single-device B=384 kernels (unsharded ShapeDtypeStructs):
            #    norms, tp_ag_res_norm (36 MiB VMEM), router_topk, dflash fp8_matmul
            S = jax.ShapeDtypeStruct
            s_u = S((b, hidden), jnp.float32)
            br_u = S((b, hidden), jnp.bfloat16)
            ag_u = S((8 * b, hidden // 8), jnp.bfloat16)
            v_u = S((hidden,), jnp.float32)

            for fn, args in (
                (
                    functools.partial(
                        rmsnorm_pallas_jax,
                        eps=1e-5,
                        has_gain=True,
                        scale_input=False,
                        out_bf16=True,
                    ),
                    (s_u, v_u),
                ),
                (
                    functools.partial(
                        residual_only_pallas_jax,
                        branch_eps=1e-8,
                        has_branch_gain=True,
                        branch_scale_input=True,
                    ),
                    (s_u, br_u, v_u, v_u, v_u),
                ),
                (
                    functools.partial(
                        residual_and_next_norm_pallas_jax,
                        branch_eps=1e-8,
                        has_branch_gain=False,
                        branch_scale_input=False,
                        next_eps=1e-5,
                        next_out_bf16=False,
                    ),
                    (s_u, br_u, v_u, v_u, v_u, v_u),
                ),
                (
                    functools.partial(
                        tp_ag_res_only_pallas_jax,
                        tp_size=8,
                        branch_eps=1e-8,
                        has_branch_gain=True,
                        branch_scale_input=True,
                    ),
                    (s_u, ag_u, v_u, v_u, v_u),
                ),
                (
                    functools.partial(
                        tp_ag_res_norm_pallas_jax,
                        tp_size=8,
                        branch_eps=1e-8,
                        has_branch_gain=True,
                        branch_scale_input=True,
                        next_eps=1e-5,
                        next_out_bf16=True,
                    ),
                    (s_u, ag_u, v_u, v_u, v_u, v_u),
                ),
                (
                    functools.partial(
                        router_topk_jax,
                        k=8,
                        renormalize=True,
                        eps=1e-15,
                        scale=2.8284271247461903,
                    ),
                    (S((b, 256), jnp.float32), S((256,), jnp.float32)),
                ),
                (
                    _fp8_matmul_jax,
                    (
                        S((b, 5120), jnp.bfloat16),
                        S((5120, 8192), jnp.float8_e4m3fn),
                        S((1, 8192), jnp.float32),
                    ),
                ),
            ):
                exp_n = export.export(jax.jit(fn), platforms=["tpu"])(*args)
                self.assertIn("stablehlo.custom_call", exp_n.mlir_module())

            # Sharded B=384 kernels: bridge_ar, twoshot_bridge_ar (6 MiB), fp8_linear_bf16
            br_sds = S((b, hidden), jnp.bfloat16, sharding=s_rep)
            op_sh = S((8 * b, hidden), jnp.bfloat16, sharding=s_mod)
            for ar_fn in (
                functools.partial(bridge_all_reduce_jax, world_size=8),
                functools.partial(bridge_twoshot_all_reduce, world_size=8),
            ):
                f_ar = jax.jit(
                    jax.shard_map(
                        ar_fn,
                        mesh=mesh8,
                        in_specs=(P("model"),),
                        out_specs=P(),
                        check_vma=False,
                    )
                )
                exp_ar = export.export(f_ar, platforms=["tpu"])(op_sh)
                self.assertIn("stablehlo.custom_call", exp_ar.mlir_module())

            f_fp8_attn = jax.jit(
                jax.shard_map(
                    fp8_linear_bf16_jax,
                    mesh=mesh8,
                    in_specs=(P(), P("model"), P("model")),
                    out_specs=P("model"),
                    check_vma=False,
                )
            )
            exp_fa = export.export(f_fp8_attn, platforms=["tpu"])(
                br_sds,
                jax.ShapeDtypeStruct((8 * hidden, 2304), jnp.float8_e4m3fn, sharding=s_mod),
                jax.ShapeDtypeStruct((8, 2304), jnp.float32, sharding=s_mod),
            )
            self.assertIn("stablehlo.custom_call", exp_fa.mlir_module())

            # 4. QKV epilogue (B=384) + prepared RPAd at dq=6 (64 seqs x 6 = 384) SWA & Full + Gate Fuse
            def _qkv_and_rpa_dq6(qkv, pos, pairs, gate, kv_cache, kv_lens, page_idx, cu_q, cu_kv, dist, *, sw):
                cs = build_decode_rope_step_tables(pos, pairs)
                q_p, kv_m = run_decode_qkv_epilogue(
                    qkv, cs, norm_eps=1e-6, query_scale=1.0, k_scale=1.0, v_scale=1.0
                )
                q_5d = q_p.reshape(1, b, 8, 2, 128)
                d_blk = (6, 4096, 6, 4096) if sw is not None else None
                return ragged_paged_attention(
                    q_5d,
                    kv_m,
                    gate,
                    kv_cache,
                    kv_lens,
                    page_idx,
                    cu_q,
                    cu_kv,
                    dist,
                    None,
                    None,
                    causal=1,
                    sm_scale=0.125,
                    sliding_window=sw,
                    k_scale=1.0,
                    v_scale=1.0,
                    decode_query_size=6,
                    d_block_sizes=d_blk,
                    _skip_mixed=True,
                    _prepared_inputs=True,
                    _fused_out_pack=2,
                    _decode_out_norm_eps=1e-6,
                    _decode_gate_in_values=True,
                )

            qkv_s = S((b, 1280), jnp.bfloat16)
            pos_s = S((b,), jnp.int32)
            pairs_s = S((8192, 128), jnp.float32)
            gate_s = S((b, 1024), jnp.bfloat16)
            kvc_s = S((4096, 128, 1, 4, 128), jnp.float8_e4m3fn)
            kvl_s = S((64,), jnp.int32)
            pidx_s = S((64 * 64,), jnp.int32)
            cuq_s = S((65,), jnp.int32)
            cukv_s = S((65,), jnp.int32)
            dist_s = S((3,), jnp.int32)

            for sw in (2049, None):
                exp_rpa = export.export(
                    jax.jit(functools.partial(_qkv_and_rpa_dq6, sw=sw)),
                    platforms=["tpu"],
                )(
                    qkv_s, pos_s, pairs_s, gate_s, kvc_s, kvl_s, pidx_s, cuq_s, cukv_s, dist_s
                )
                self.assertIn("stablehlo.custom_call", exp_rpa.mlir_module())

            # 5. Fused FP8 LM head argmax (k=1, B=320 & B=384) + Top-64 (B=384) + lossless VMEM-AG Top-64
            from sglang_tpu.kernels.fp8_lm_head_fused import (
                distributed_topk_vmem_ag_jax,
                fp8_lm_head_argmax_jax,
                fp8_lm_head_fused_topk64_jax,
            )

            v_loc = 25256
            vp = 25600
            lw_s = S((8 * hidden, vp), jnp.float8_e4m3fn, sharding=s_mod)
            ls_s = S((8, vp), jnp.float32, sharding=s_mod)
            for b_lm in (320, 384):
                xl_s = S((b_lm, hidden), jnp.bfloat16, sharding=s_rep)
                f_k1 = jax.jit(
                    jax.shard_map(
                        functools.partial(
                            fp8_lm_head_argmax_jax,
                            world_size=8,
                            local_width=v_loc,
                            org_vocab_size=200000,
                            global_vocab_size=202048,
                            org_vocab_padded_count=25000,
                        ),
                        mesh=mesh8,
                        in_specs=(P(), P("model"), P("model")),
                        out_specs=(P(), P()),
                        check_vma=False,
                    )
                )
                exp_k1 = export.export(f_k1, platforms=["tpu"])(xl_s, lw_s, ls_s)
                self.assertIn("stablehlo.custom_call", exp_k1.mlir_module())

            f_k64 = jax.jit(
                jax.shard_map(
                    functools.partial(
                        fp8_lm_head_fused_topk64_jax,
                        world_size=8,
                        local_width=v_loc,
                        org_vocab_size=200000,
                        global_vocab_size=202048,
                        org_vocab_padded_count=25000,
                        k=64,
                    ),
                    mesh=mesh8,
                    in_specs=(P(), P("model"), P("model")),
                    out_specs=(P(), P()),
                    check_vma=False,
                )
            )
            exp_k64 = export.export(f_k64, platforms=["tpu"])(br_sds, lw_s, ls_s)
            self.assertIn("stablehlo.custom_call", exp_k64.mlir_module())

            f_vmem_tk = jax.jit(
                jax.shard_map(
                    functools.partial(
                        distributed_topk_vmem_ag_jax,
                        world_size=8,
                        local_width=v_loc,
                        org_vocab_size=200000,
                        global_vocab_size=202048,
                        org_vocab_padded_count=25000,
                        k=64,
                    ),
                    mesh=mesh8,
                    in_specs=(P("model"),),
                    out_specs=(P(), P()),
                    check_vma=False,
                )
            )
            exp_vtk = export.export(f_vmem_tk, platforms=["tpu"])(
                S((8 * b, v_loc), jnp.float32, sharding=s_mod)
            )
            self.assertIn("stablehlo.custom_call", exp_vtk.mlir_module())


def dataclasses_replace_stat(cfg):
    import dataclasses

    return dataclasses.replace(cfg, static_holder_mask=True)


if __name__ == "__main__":
    unittest.main()
