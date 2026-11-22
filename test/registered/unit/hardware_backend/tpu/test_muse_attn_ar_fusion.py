# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests and TPU7x-target Mosaic export checks for Muse attention QKVG + o_proj AR fusion."""

from __future__ import annotations

import functools
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.models.muse_spark import (
    MuseSparkAttention,
    MuseSparkDecoderLayer,
    MuseSparkGatedResidual,
    MuseSparkRMSNorm,
)
from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
    TPU7X_CHIPMATE_XOR,
    bridge_all_reduce_jax,
    fp8_linear_bf16_jax,
    quantize_fp8_k_major,
    tp_ar_res_norm_jax,
    tp_attn_tail_ar_res_norm_jax,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_mock_attention(
    *,
    hidden: int = 512,
    num_heads: int = 2,
    num_kv_heads: int = 1,
    head_dim: int = 128,
    tp_size: int = 8,
    tp_rank: int = 0,
) -> MuseSparkAttention:
    attn = MuseSparkAttention.__new__(MuseSparkAttention)
    nn.Module.__init__(attn)
    attn.head_dim = head_dim
    attn.num_heads = num_heads
    attn.num_kv_heads = num_kv_heads
    attn.q_size = num_heads * head_dim
    attn.kv_size = num_kv_heads * head_dim
    attn.norm_eps = 1e-5
    attn.decode_qkv_epilogue = False
    attn.query_scale = 1.0
    attn.rotary_emb = None
    attn._fp8_attn_proj = False
    for bname in (
        "_qkvg_weight",
        "_qkvg_w8k",
        "_qkvg_scale",
        "_o_proj_wk",
        "_o_proj_w8k",
        "_o_proj_scale",
    ):
        attn.register_buffer(bname, None, persistent=False)

    torch.manual_seed(1119)
    qkv_dim = attn.q_size + 2 * attn.kv_size
    w_qkv = (torch.randn(qkv_dim, hidden) * 0.05).to(torch.bfloat16)
    w_gate = (torch.randn(attn.q_size, hidden) * 0.05).to(torch.bfloat16)
    w_o = (torch.randn(hidden, attn.q_size) * 0.05).to(torch.bfloat16)

    class _FakeColLinear(nn.Module):
        def __init__(self, w: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(w, requires_grad=False)
            self.bias = None
            self.gather_output = False
            self.tp_size = tp_size
            self.tp_rank = tp_rank

        def forward(self, x: torch.Tensor):
            return torch.nn.functional.linear(x, self.weight), None

    class _FakeRowLinear(nn.Module):
        def __init__(self, w: torch.Tensor):
            super().__init__()
            self.weight = nn.Parameter(w, requires_grad=False)
            self.bias = None
            self.reduce_results = True
            self.tp_size = tp_size
            self.tp_rank = tp_rank
            self.output_size = w.shape[0]
            self.use_dp_attention_reduce = False
            self.use_decode_attn_tp = False
            self.last_skip_ar: bool | None = None

        def forward(self, x: torch.Tensor, skip_all_reduce: bool = False):
            self.last_skip_ar = skip_all_reduce
            return torch.nn.functional.linear(x, self.weight), None

    attn.qkv_proj = _FakeColLinear(w_qkv)
    attn.gate_proj = _FakeColLinear(w_gate)
    attn.o_proj = _FakeRowLinear(w_o)
    # Deterministic attention mock combining q, k, v so every projection slice is tested.
    attn.attn = lambda q, k, v, fb: (
        q + k.repeat(1, num_heads // num_kv_heads) * 0.5 - v.repeat(1, num_heads // num_kv_heads) * 0.25
    ).to(q.dtype)
    return attn


class TestMuseAttnArFusion(CustomTestCase):
    def test_new_env_flags_default_off(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in (
                "SGLANG_TPU_MUSE_FUSE_QKVG_PROJ",
                "SGLANG_TPU_MUSE_DECODE_ATTN_QKV_TPAR",
                "SGLANG_TPU_MUSE_DECODE_FP8_ATTN_PROJ",
                "SGLANG_TPU_MUSE_DECODE_FUSED_O_PROJ_AR",
            ):
                os.environ.pop(k, None)
            self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_FUSE_QKVG_PROJ.get())
            self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_DECODE_ATTN_QKV_TPAR.get())
            self.assertFalse(tpu_envs.SGLANG_TPU_MUSE_DECODE_FP8_ATTN_PROJ.get())
            self.assertEqual(tpu_envs.SGLANG_TPU_MUSE_DECODE_FUSED_O_PROJ_AR.get(), 0)
            self.assertEqual(TPU7X_CHIPMATE_XOR, 1)

    def test_fused_qkvg_parity_and_storage_release(self):
        attn = _make_mock_attention()
        x = (torch.randn(32, 512) * 0.5).to(torch.bfloat16)
        pos = torch.arange(32, dtype=torch.int32)
        fb_dec = SimpleNamespace(forward_mode=ForwardMode.DECODE, attn_backend=None)
        fb_ext = SimpleNamespace(forward_mode=ForwardMode.EXTEND, attn_backend=None)

        ref_dec = attn(pos, x, fb_dec)
        ref_ext = attn(pos, x, fb_ext)

        ok = attn.prepare_fused_qkvg(fp8_wo=False)
        self.assertTrue(ok)
        self.assertIsNotNone(attn._qkvg_weight)
        self.assertEqual(tuple(attn._qkvg_weight.shape), (512 + 256, 512))
        self.assertTrue(attn._qkvg_weight.is_contiguous())
        # Original separate weights must be emptied (zero net extra HBM).
        self.assertEqual(attn.qkv_proj.weight.numel(), 0)
        self.assertEqual(attn.gate_proj.weight.numel(), 0)

        got_dec = attn(pos, x, fb_dec)
        got_ext = attn(pos, x, fb_ext)
        torch.testing.assert_close(got_dec, ref_dec, rtol=0.0, atol=0.0)
        torch.testing.assert_close(got_ext, ref_ext, rtol=0.0, atol=0.0)

        # Also check return_attn_and_gate hook for mode=3
        a_raw, gate = attn(pos, x, fb_dec, return_attn_and_gate=True)
        self.assertEqual(tuple(a_raw.shape), (32, 256))
        self.assertEqual(tuple(gate.shape), (32, 256))

        # And skip_o_proj_all_reduce hook for mode=1/2
        _ = attn(pos, x, fb_dec, skip_o_proj_all_reduce=True)
        self.assertTrue(attn.o_proj.last_skip_ar)

    def test_quantize_fp8_k_major_accuracy(self):
        torch.manual_seed(7)
        w = (torch.randn(2304, 1024) * 0.03).to(torch.bfloat16)
        w8k, scale = quantize_fp8_k_major(w)
        self.assertEqual(w8k.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(w8k.shape), (1024, 2304))
        self.assertEqual(tuple(scale.shape), (1, 2304))
        self.assertTrue(w8k.is_contiguous())
        deq = (w8k.float() * scale).t()
        cos = torch.nn.functional.cosine_similarity(
            deq.flatten(), w.float().flatten(), dim=0
        ).item()
        self.assertGreater(cos, 0.998)

    def test_decoder_layer_fused_o_proj_ar_dispatch_and_fallbacks(self):
        from sglang.srt.layers.linear import RowParallelLinear

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
        o_proj.weight = nn.Parameter(
            torch.zeros((hidden, 256), dtype=torch.bfloat16), requires_grad=False
        )

        attn_calls = []

        class _StubAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.o_proj = o_proj
                self.head_dim = 128
                self.norm_eps = 1e-5
                self._o_proj_wk = torch.zeros((256, hidden), dtype=torch.bfloat16)
                self._o_proj_w8k = None
                self._o_proj_scale = None

            def forward(
                self,
                positions,
                hs,
                fb,
                *,
                skip_o_proj_all_reduce=False,
                return_attn_and_gate=False,
            ):
                attn_calls.append((skip_o_proj_all_reduce, return_attn_and_gate))
                if return_attn_and_gate:
                    z = torch.zeros((hs.shape[0], 256), dtype=torch.bfloat16)
                    return z, z
                return torch.zeros((hs.shape[0], hidden), dtype=torch.bfloat16)

        layer.self_attn = _StubAttn()
        layer.mlp = Mock(
            tp_size=8,
            side_effect=lambda h, r, **kw: torch.zeros_like(h),
        )
        layer.register_buffer("_dummy_gain", torch.ones(hidden, dtype=torch.float32))

        fake_tpar = Mock(
            side_effect=lambda s, op, a, b, ng, be, ne: (
                s.clone(),
                s.clone(),
                s.to(torch.bfloat16),
            )
        )
        fake_tail = Mock(
            side_effect=lambda s, ar, g, wk, a, b, ng, he, be, ne: (
                s.clone(),
                s.clone(),
                s.to(torch.bfloat16),
            )
        )
        fake_rmsnorm = Mock(
            side_effect=lambda x, g, eps, hg, si, bf16: (
                x.to(torch.bfloat16) if bf16 else x.float()
            )
        )
        fake_res_only = Mock(side_effect=lambda s, *a: s.clone())

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "8"}),
            patch(
                "sglang.srt.runtime_context.get_forward",
                return_value=SimpleNamespace(sp_active=False),
            ),
            patch(
                "sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm.get_muse_decode_tp_ar_res_norm_op",
                return_value=fake_tpar,
            ),
            patch(
                "sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm.get_muse_decode_attn_tail_ar_res_norm_op",
                return_value=fake_tail,
            ),
            patch(
                "sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router.get_muse_decode_rmsnorm_op",
                return_value=fake_rmsnorm,
            ),
            patch(
                "sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router.get_muse_decode_res_only_op",
                return_value=fake_res_only,
            ),
        ):
            pos = torch.arange(64, dtype=torch.int32)
            stream = torch.randn(64, hidden, dtype=torch.float32)

            # Mode 1 + norm_layout=3 on DECODE and TARGET_VERIFY fires fake_tpar
            layer._decode_norm_layout = 3
            layer._decode_fused_o_proj_ar = 1
            for fm in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                attn_calls.clear()
                fake_tpar.reset_mock()
                _ = layer(pos, stream, SimpleNamespace(forward_mode=fm))
                self.assertEqual(fake_tpar.call_count, 1)
                self.assertEqual(attn_calls, [(True, False)])

            # EXTEND and odd batch B=8 must safely bypass fused o_proj AR
            fake_tpar.reset_mock()
            _ = layer(pos, stream, SimpleNamespace(forward_mode=ForwardMode.EXTEND))
            self.assertEqual(fake_tpar.call_count, 0)
            self.assertEqual(
                layer._active_fused_o_proj_ar_mode(
                    8, SimpleNamespace(forward_mode=ForwardMode.DECODE)
                ),
                0,
            )

            # Mode 2 + norm_layout=3 fires standalone bridge_all_reduce + nl5 res_norm
            fake_bar = Mock(side_effect=lambda op: op.clone())
            fake_res_norm = Mock(side_effect=lambda s, *a: (s.clone(), s.clone()))
            with (
                patch(
                    "sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm.get_muse_decode_bridge_ar_op",
                    return_value=fake_bar,
                ),
                patch(
                    "sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router.get_muse_decode_res_norm_op",
                    return_value=fake_res_norm,
                ),
            ):
                layer._decode_norm_layout = 3
                layer._decode_fused_o_proj_ar = 2
                attn_calls.clear()
                _ = layer(pos, stream, SimpleNamespace(forward_mode=ForwardMode.DECODE))
                self.assertEqual(fake_bar.call_count, 1)
                self.assertEqual(fake_res_norm.call_count, 1)
                self.assertEqual(attn_calls, [(True, False)])

            # Mode 3 + norm_layout=0 fires 8-op fake_tail cleanly
            layer._decode_norm_layout = 0
            layer._decode_fused_o_proj_ar = 3
            attn_calls.clear()
            fake_tail.reset_mock()
            _ = layer(pos, stream, SimpleNamespace(forward_mode=ForwardMode.DECODE))
            self.assertEqual(fake_tail.call_count, 1)
            self.assertEqual(attn_calls, [(False, True)])

            # Mode 1 and Mode 3 gate off at B=256 while Mode 2 stays active on TP8
            for mode, want_active in ((1, 0), (2, 2), (3, 0)):
                layer._decode_fused_o_proj_ar = mode
                self.assertEqual(
                    layer._active_fused_o_proj_ar_mode(
                        256, SimpleNamespace(forward_mode=ForwardMode.DECODE)
                    ),
                    want_active,
                )

    def test_tpu7x_mosaic_export_all_variants(self):
        import jax
        import jax.numpy as jnp
        from jax import export
        from jax.sharding import AbstractMesh, NamedSharding, PartitionSpec as P
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        devs = jax.devices("cpu")
        if len(devs) < 8:
            self.skipTest("Requires XLA_FLAGS=--xla_force_host_platform_device_count>=8")

        tp_list = [8, 16] if len(devs) >= 16 else [8]
        hidden = 8192
        for tp in tp_list:
            q_loc = 8192 // tp
            qkvg_loc = q_loc + 2 * max(128, 1024 // tp) + q_loc
            amesh = AbstractMesh((tp,), ("model",))
            s_rep = NamedSharding(amesh, P())
            s_mod = NamedSharding(amesh, P("model"))

            for b in (16, 32, 64, 128, 256):
                with (
                    self.subTest(tp=tp, b=b),
                    tpu_info_for_cpu_export(),
                    jax.sharding.use_abstract_mesh(amesh),
                ):
                    s_sds = jax.ShapeDtypeStruct((b, hidden), jnp.float32, sharding=s_rep)
                    op_sds = jax.ShapeDtypeStruct((tp * b, hidden), jnp.bfloat16, sharding=s_mod)
                    v_sds = jax.ShapeDtypeStruct((1, hidden), jnp.float32, sharding=s_rep)

                    # 1. tp_ar_res_norm (mode=1)
                    f1 = jax.jit(
                        jax.shard_map(
                            functools.partial(
                                tp_ar_res_norm_jax,
                                world_size=tp,
                                branch_eps=1e-8,
                                next_eps=1e-5,
                            ),
                            in_specs=(P(), P("model"), P(), P(), P()),
                            out_specs=(P(), P(), P()),
                            check_vma=False,
                        )
                    )
                    exp1 = export.export(f1, platforms=["tpu"])(
                        s_sds, op_sds, v_sds, v_sds, v_sds
                    )
                    self.assertIn("stablehlo.custom_call", exp1.mlir_module())

                    # 2. bridge_all_reduce (mode=2) & 8-op tail BF16/FP8 (mode=3) + FP8 QKVG
                    if b in (16, 64, 256):
                        f2 = jax.jit(
                            jax.shard_map(
                                functools.partial(bridge_all_reduce_jax, world_size=tp),
                                in_specs=(P("model"),),
                                out_specs=P(),
                                check_vma=False,
                            )
                        )
                        exp2 = export.export(f2, platforms=["tpu"])(op_sds)
                        self.assertIn("stablehlo.custom_call", exp2.mlir_module())

                        a_sds = jax.ShapeDtypeStruct((tp * b, q_loc), jnp.bfloat16, sharding=s_mod)
                        wo_bf16 = jax.ShapeDtypeStruct((tp * q_loc, hidden), jnp.bfloat16, sharding=s_mod)
                        wo_fp8 = jax.ShapeDtypeStruct((tp * q_loc, hidden), jnp.float8_e4m3fn, sharding=s_mod)
                        ws_sds = jax.ShapeDtypeStruct((tp, hidden), jnp.float32, sharding=s_mod)

                        f3 = jax.jit(
                            jax.shard_map(
                                lambda s, a, g, wk, al, be, ng: tp_attn_tail_ar_res_norm_jax(
                                    s, a, g, wk, None, al, be, ng, world_size=tp
                                ),
                                in_specs=(P(), P("model"), P("model"), P("model"), P(), P(), P()),
                                out_specs=(P(), P(), P()),
                                check_vma=False,
                            )
                        )
                        exp3 = export.export(f3, platforms=["tpu"])(
                            s_sds, a_sds, a_sds, wo_bf16, v_sds, v_sds, v_sds
                        )
                        self.assertIn("stablehlo.custom_call", exp3.mlir_module())

                        f3_fp8 = jax.jit(
                            jax.shard_map(
                                lambda s, a, g, wk, ws, al, be, ng: tp_attn_tail_ar_res_norm_jax(
                                    s, a, g, wk, ws, al, be, ng, world_size=tp
                                ),
                                in_specs=(P(), P("model"), P("model"), P("model"), P("model"), P(), P(), P()),
                                out_specs=(P(), P(), P()),
                                check_vma=False,
                            )
                        )
                        exp3f = export.export(f3_fp8, platforms=["tpu"])(
                            s_sds, a_sds, a_sds, wo_fp8, ws_sds, v_sds, v_sds, v_sds
                        )
                        self.assertIn("stablehlo.custom_call", exp3f.mlir_module())

                        x_sds = jax.ShapeDtypeStruct((b, hidden), jnp.bfloat16, sharding=s_rep)
                        q8_sds = jax.ShapeDtypeStruct((tp * hidden, qkvg_loc), jnp.float8_e4m3fn, sharding=s_mod)
                        qs_sds = jax.ShapeDtypeStruct((tp, qkvg_loc), jnp.float32, sharding=s_mod)
                        f4 = jax.jit(
                            jax.shard_map(
                                fp8_linear_bf16_jax,
                                in_specs=(P(), P("model"), P("model")),
                                out_specs=P("model"),
                                check_vma=False,
                            )
                        )
                        exp4 = export.export(f4, platforms=["tpu"])(
                            x_sds, q8_sds, qs_sds
                        )
                        self.assertIn("stablehlo.custom_call", exp4.mlir_module())


if __name__ == "__main__":
    unittest.main()
