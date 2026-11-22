# SPDX-License-Identifier: Apache-2.0
"""Unit, dispatch, cold-compile, and 16-device TPU7x Mosaic export tests for Wave-7 bridges."""

from __future__ import annotations

import ast
import inspect
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _make_stub_parallel(tp_size: int = 8):
    par = types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=0,
        moe_ep_size=tp_size,
        moe_ep_rank=0,
        moe_tp_size=1,
        pp_group=types.SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    return par


def _make_fake_linear(out_f: int, in_f: int, *, tp_size: int = 8, is_row: bool = False):
    from sglang_tpu.models.muse_spark import RowParallelLinear

    class _FakeRow(RowParallelLinear):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(
                torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.02, requires_grad=False
            )
            self.bias = None
            self.tp_size = tp_size
            self.reduce_results = True
            self.output_size = out_f
            self.use_dp_attention_reduce = False
            self.use_decode_attn_tp = False
            self.calls = 0

        def forward(self, x, skip_all_reduce: bool = False):
            self.calls += 1
            return torch.matmul(x, self.weight.t()), None

    class _FakeCol(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(out_f, in_f, dtype=torch.bfloat16) * 0.02, requires_grad=False
            )
            self.bias = None
            self.tp_size = tp_size
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            return torch.matmul(x, self.weight.t().to(x.dtype)), None

    return _FakeRow() if is_row else _FakeCol()


def _make_norm(dim: int, *, has_weight: bool = True, scale_input: bool = False):
    from sglang_tpu.models.muse_spark import MuseSparkRMSNorm

    return MuseSparkRMSNorm(dim, eps=1e-5, has_weight=has_weight, scale_input=scale_input)


def _make_res_gate(dim: int):
    from sglang_tpu.models.muse_spark import MuseSparkGatedResidual

    return MuseSparkGatedResidual(dim, temperature=1.0)


def _make_fake_fb(mode_name: str = "DECODE"):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    fm = getattr(ForwardMode, mode_name)
    return types.SimpleNamespace(
        forward_mode=fm,
        return_logprob=False,
        capture_hidden_mode=None,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
        extend_seq_lens=None,
    )


def test_w7_env_defaults_and_ast_no_envfield_bool():
    """All Wave-7 collective-bridge flags must default off and never trigger EnvField.__bool__."""
    from sglang_tpu.envs import envs

    assert envs.SGLANG_TPU_MUSE_DECODE_COLLECTIVE_BRIDGES.default == 0
    assert envs.SGLANG_TPU_MUSE_DECODE_BRIDGE_ROUTER_BF16.default is False
    assert envs.SGLANG_TPU_MUSE_DECODE_BRIDGE_FREE_BF16.default is False

    # Verify our modified files never use `tpu_envs.SGLANG_*` directly in a boolean test.
    for rel in (
        "sglang_tpu/kernels/decode_collective_bridges.py",
        "sglang_tpu/models/muse_spark.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/attn_ar_res_norm.py",
    ):
        tree = ast.parse((_REPO_ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                test_src = ast.unparse(node.test)
                assert "tpu_envs.SGLANG_TPU_MUSE_DECODE_COLLECTIVE" not in test_src or ".get()" in test_src
                assert "tpu_envs.SGLANG_TPU_MUSE_DECODE_BRIDGE_" not in test_src or ".get()" in test_src


def test_quantize_weight_fp8_k_major_accuracy_and_contiguity():
    """Verify K-major FP8 weight quantization accuracy, contiguity, and shape checks."""
    from sglang_tpu.kernels.decode_collective_bridges import quantize_weight_fp8_k_major

    torch.manual_seed(0)
    w = torch.randn(1024, 512, dtype=torch.bfloat16) * 0.03
    w8k, sc = quantize_weight_fp8_k_major(w)
    assert w8k.shape == (512, 1024)
    assert w8k.dtype == torch.float8_e4m3fn
    assert w8k.is_contiguous()
    assert sc.shape == (1, 1024)
    assert sc.dtype == torch.float32
    assert sc.is_contiguous()

    recon = (w8k.float() * sc).t()
    cos = torch.nn.functional.cosine_similarity(
        w.float().reshape(1, -1), recon.reshape(1, -1)
    ).item()
    assert cos > 0.9994

    with pytest.raises(ValueError):
        quantize_weight_fp8_k_major(torch.randn(128, 512, dtype=torch.bfloat16))


def test_attn_ar_res_norm_static_arg_in_specs_fix():
    """Regression test for #1098 §2: input_partition_specs must pad trailing None for static floats."""
    from sglang_tpu.moe.pallas.fused_rs_muse import attn_ar_res_norm as arn

    captured = {}

    def _fake_export(name, fn, *, mesh=None, input_partition_specs=None, output_partition_specs=None):
        sig = inspect.signature(fn)
        captured[name] = (len(sig.parameters), input_partition_specs, output_partition_specs)
        return fn

    fake_mesh = object()
    with (
        mock.patch.object(arn, "_get_tp_mesh", return_value=fake_mesh),
        mock.patch("sglang_tpu.pallas_jax_op.cpu_export_jax_op", side_effect=_fake_export),
        mock.patch.dict(arn._TP_AR_RES_NORM_OPS, {}, clear=True),
        mock.patch.dict(arn._ATTN_TAIL_FP8_OPS, {}, clear=True),
        mock.patch.dict(arn._ATTN_TAIL_BF16_OPS, {}, clear=True),
    ):
        arn.get_muse_decode_tp_ar_res_norm_op(8)
        arn.get_muse_decode_attn_tail_ar_res_norm_op(8, fp8_wo=True)
        arn.get_muse_decode_attn_tail_ar_res_norm_op(8, fp8_wo=False)

    assert len(captured) == 3
    for op_name, (n_params, in_specs, _out_specs) in captured.items():
        assert len(in_specs) == n_params, f"{op_name}: {len(in_specs)} != {n_params}"
        assert in_specs[-1] is None and in_specs[-2] is None


def test_prewarm_lock_free_cold_compile_safety():
    """Verify prewarm registers all bridge ops so cold torch.compile never enters _OPS_LOCK."""
    from sglang_tpu.kernels import decode_collective_bridges as dcb
    from sglang_tpu.moe.pallas.fused_rs_muse import attn_ar_res_norm as arn

    class _ExplodingLock:
        def __enter__(self):
            raise AssertionError("_OPS_LOCK entered during compiled forward!")

        def __exit__(self, *args):
            return False

    def _fake_export(name, fn, **kw):
        fn.register_fake = lambda f: f
        return fn

    fake_mesh = object()
    with (
        mock.patch.object(dcb, "_get_tp_mesh", return_value=fake_mesh),
        mock.patch("sglang_tpu.pallas_jax_op.cpu_export_jax_op", side_effect=_fake_export),
        mock.patch.dict(dcb._ATTN_TO_MOE_OPS, {}, clear=True),
        mock.patch.dict(dcb._POST_ATTN_TO_MOE_OPS, {}, clear=True),
        mock.patch.dict(dcb._MOE_TO_ATTN_QKVG_OPS, {}, clear=True),
        mock.patch.dict(dcb._MOE_POST_AG_RN_OPS, {}, clear=True),
        mock.patch.dict(dcb._MOE_POST_AG_RO_OPS, {}, clear=True),
        mock.patch.object(dcb, "_DUAL_QKVG_OP", None),
        mock.patch.object(arn, "_FP8_LINEAR_BF16_OP", []),
    ):
        # Verify mode=3 ALONE warms arn.get_muse_decode_fp8_linear_bf16_op() (#1215 Bug #1)!
        dcb.prewarm_decode_collective_bridge_ops(8, mode=3, bf16_router=False)
        with (
            mock.patch.object(dcb, "_OPS_LOCK", _ExplodingLock()),
            mock.patch.object(arn, "_OPS_LOCK", _ExplodingLock()),
        ):
            assert callable(arn.get_muse_decode_fp8_linear_bf16_op())
            assert callable(dcb.get_attn_to_moe_bridge_op(8, bf16_router=False))

        dcb.prewarm_decode_collective_bridge_ops(8, mode=3, bf16_router=True)
        dcb.prewarm_decode_collective_bridge_ops(8, mode=5)
        with (
            mock.patch.object(dcb, "_OPS_LOCK", _ExplodingLock()),
            mock.patch.object(arn, "_OPS_LOCK", _ExplodingLock()),
        ):
            assert callable(dcb.get_attn_to_moe_bridge_op(8, bf16_router=False))
            assert callable(dcb.get_attn_to_moe_bridge_op(8, bf16_router=True))
            assert callable(dcb.get_post_attn_to_moe_bridge_op(8))
            assert callable(dcb.get_moe_to_attn_bridge_op(8, sharded_moe_in=False))
            assert callable(dcb.get_moe_to_attn_bridge_op(8, sharded_moe_in=True))
            assert callable(dcb.get_moe_post_ag_res_norm_op(8, sharded_moe_in=False))
            assert callable(dcb.get_moe_post_ag_res_norm_op(8, sharded_moe_in=True))
            assert callable(dcb.get_moe_post_ag_res_only_op(8, sharded_moe_in=False))
            assert callable(dcb.get_moe_post_ag_res_only_op(8, sharded_moe_in=True))
            assert callable(dcb.get_fp8_dual_qkv_gate_op())
            assert callable(arn.get_muse_decode_fp8_linear_bf16_op())


def test_moe_free_bf16_fallback_all_gathers_when_gather_output_true():
    """Regression test for #1215 Bug #2: w7_wpre/w7_wpost fallback must all_gather when gather_output=True."""
    layer = _make_dummy_layer(0, H=512, H_PRE=256, tp_size=8)
    mlp = layer.mlp
    # Simulate DECODE_ONESHOT_MOE_AG=0: pre_expert_proj=[32, 512], post_expert_proj=[64, 256], gather_output=True
    mlp.pre_expert_proj = _make_fake_linear(256, 512, tp_size=8, is_row=False)
    mlp.pre_expert_proj.gather_output = True
    mlp.post_expert_proj = _make_fake_linear(256, 256 * 8, tp_size=8, is_row=False)
    mlp.post_expert_proj.gather_output = True
    mlp.oneshot_gather = False
    mlp._manual_post_gather = False
    mlp.pre_expert_norm = _make_norm(256 * 8)

    pre_ok, post_ok = mlp.prepare_w7_moe_bridges(
        do_pre=True, do_post=True, router_bf16=False, free_bf16=True
    )
    assert pre_ok and post_ok
    assert mlp.pre_expert_proj.weight.shape[0] == 0
    assert mlp.post_expert_proj.weight.shape[0] == 0

    ag_calls = []

    def _fake_tp_ag(t, *args, **kwargs):
        ag_calls.append(tuple(t.shape))
        return t.repeat(1, 8)

    hs = torch.randn(12, 512, dtype=torch.bfloat16)
    ri = hs.float()
    with mock.patch(
        "sglang_tpu.models.muse_spark.tensor_model_parallel_all_gather",
        side_effect=_fake_tp_ag,
    ):
        out = mlp(hs, ri, decode=False)
    assert ag_calls == [(12, 256), (12, 256)]
    assert out.shape == (12, 256 * 8)


def test_attention_w7_prepare_free_bf16_and_head_dim64_return_gated():
    """Verify MuseSparkAttention handles head_dim=64 return_gated_attn + precomputed_qkv_gate + free_bf16."""
    from sglang_tpu.models.muse_spark import MuseSparkAttention

    H, Q_COL, KV_COL = 512, 256, 128
    QKV_COL = Q_COL + 2 * KV_COL

    attn = object.__new__(MuseSparkAttention)
    torch.nn.Module.__init__(attn)
    attn.q_size = Q_COL
    attn.kv_size = KV_COL
    attn.head_dim = 64  # Actual Muse Spark head_dim (never gated on 128!)
    attn.norm_eps = 1e-5
    attn.query_scale = 1.0
    attn.rotary_emb = None
    attn.decode_qkv_epilogue = False
    attn.decode_attn_gate_fuse = False
    attn._qkvg_weight = None
    attn._qkvg_w8k = None
    attn._qkvg_scale = None
    attn._o_proj_w8k = None
    attn._o_proj_scale = None
    attn._o_proj_wk = None
    attn.qkv_proj = _make_fake_linear(QKV_COL, H, tp_size=8, is_row=False)
    attn.gate_proj = _make_fake_linear(Q_COL, H, tp_size=8, is_row=False)
    attn.o_proj = _make_fake_linear(H, Q_COL, tp_size=8, is_row=True)
    attn.attn = lambda q, k, v, fb: q

    assert attn.prepare_w7_qkv_gate_fp8(free_bf16=True) is True
    assert attn.prepare_w7_o_proj_fp8(free_bf16=True) is True
    assert attn.qkv_proj.weight.shape[0] == 0
    assert attn.gate_proj.weight.shape[0] == 0
    assert attn.o_proj.weight.shape[0] == 0

    B = 16
    pos = torch.arange(B, dtype=torch.int32)
    qkv_in = torch.randn(B, QKV_COL, dtype=torch.bfloat16)
    gate_in = torch.randn(B, Q_COL, dtype=torch.bfloat16)
    fb_dec = _make_fake_fb("DECODE")

    # precomputed_qkv_gate + return_gated_attn=True skips qkv_proj, gate_proj, AND o_proj!
    gated = attn(
        pos,
        None,
        fb_dec,
        precomputed_qkv_gate=(qkv_in, gate_in),
        return_gated_attn=True,
    )
    assert gated.shape == (B, Q_COL)
    assert attn.qkv_proj.calls == 0 and attn.gate_proj.calls == 0 and attn.o_proj.calls == 0

    # Short EXTEND fallback after free_bf16=True still works cleanly via FP8 buffers!
    fb_ext = _make_fake_fb("EXTEND")
    x_ext = torch.randn(64, H, dtype=torch.bfloat16)
    dual_calls = []

    def _fake_dual(x, wq, sq, wg, sg):
        dual_calls.append(x.shape[0])
        return (
            (torch.matmul(x, wq.to(x.dtype)).float() * sq).to(x.dtype),
            (torch.matmul(x, wg.to(x.dtype)).float() * sg).to(x.dtype),
        )

    def _fake_fp8_lin(x, w, s):
        return (torch.matmul(x, w.to(x.dtype)).float() * s).to(x.dtype)

    with (
        mock.patch(
            "sglang_tpu.kernels.decode_collective_bridges.get_fp8_dual_qkv_gate_op",
            return_value=_fake_dual,
        ),
        mock.patch(
            "sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm.get_muse_decode_fp8_linear_bf16_op",
            return_value=_fake_fp8_lin,
        ),
        mock.patch(
            "sglang_tpu.models.muse_spark.tensor_model_parallel_all_reduce",
            side_effect=lambda t: t,
        ),
    ):
        out_ext = attn(pos, x_ext, fb_ext)
    assert out_ext.shape == (64, H)
    assert dual_calls == [64]


def _make_dummy_layer(mode: int, *, H: int = 512, H_PRE: int = 256, tp_size: int = 8):
    from sglang_tpu.models.muse_spark import MuseSparkAttention, MuseSparkDecoderLayer, MuseSparkMoE

    Q_COL, KV_COL = 256, 128
    QKV_COL = Q_COL + 2 * KV_COL

    layer = object.__new__(MuseSparkDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.input_layernorm = _make_norm(H)
    layer.post_attention_layernorm = _make_norm(H, has_weight=False)
    layer.post_attention_residual_gate = _make_res_gate(H)
    layer.pre_feedforward_layernorm = _make_norm(H)
    layer.post_feedforward_layernorm = _make_norm(H, scale_input=True)
    layer.post_feedforward_residual_gate = _make_res_gate(H)
    layer._decode_norm_layout = 0
    layer._decode_fuse_target_verify = False
    layer._decode_fused_o_proj_ar = 0
    layer._decode_col_bridges = mode

    attn = object.__new__(MuseSparkAttention)
    torch.nn.Module.__init__(attn)
    attn.q_size = Q_COL
    attn.kv_size = KV_COL
    attn.head_dim = 64
    attn.norm_eps = 1e-5
    attn.query_scale = 1.0
    attn.rotary_emb = None
    attn.decode_qkv_epilogue = False
    attn.decode_attn_gate_fuse = False
    attn._qkvg_weight = None
    attn._qkvg_w8k = None
    attn._qkvg_scale = None
    attn._o_proj_w8k = None
    attn._o_proj_scale = None
    attn._o_proj_wk = None
    attn.qkv_proj = _make_fake_linear(QKV_COL, H, tp_size=tp_size, is_row=False)
    attn.gate_proj = _make_fake_linear(Q_COL, H, tp_size=tp_size, is_row=False)
    attn.o_proj = _make_fake_linear(H, Q_COL, tp_size=tp_size, is_row=True)
    attn.attn = lambda q, k, v, fb: q
    layer.self_attn = attn

    mlp = object.__new__(MuseSparkMoE)
    torch.nn.Module.__init__(mlp)
    mlp.tp_size = tp_size
    mlp.moe_tp_size = 1
    mlp.top_k = 8
    mlp.router_multiplier = 0.25
    mlp.route_eps = 1e-5
    mlp.route_scale = 1.0
    mlp._decode_fused_router = False
    mlp._fused_decode_op = None
    mlp.oneshot_gather = False
    mlp._manual_post_gather = False
    mlp.gate = _make_fake_linear(256, H, tp_size=1, is_row=False)
    mlp.e_score_correction_bias = torch.nn.Parameter(torch.zeros(256), requires_grad=False)
    mlp.pre_expert_proj = _make_fake_linear(H_PRE, H, tp_size=tp_size, is_row=False)
    mlp.pre_expert_norm = _make_norm(H_PRE)
    mlp.post_expert_proj = _make_fake_linear(H, H_PRE, tp_size=tp_size, is_row=False)

    class _FakeExperts(torch.nn.Module):
        handles_moe_reduction = True

        def forward(self, x, topk):
            return x * 0.5

    mlp.experts = _FakeExperts()
    layer.mlp = mlp

    if mode > 0:
        if mode in (1, 3, 4, 5, 6):
            assert attn.prepare_w7_o_proj_fp8()
        if mode in (2, 3, 5):
            assert attn.prepare_w7_qkv_gate_fp8()
        pre_ok, post_ok = mlp.prepare_w7_moe_bridges(
            do_pre=mode in (1, 3, 4, 5, 6),
            do_post=mode in (2, 3, 4, 5, 6),
            router_bf16=True,
        )
        assert pre_ok == (mode in (1, 3, 4, 5, 6))
        assert post_ok == (mode in (2, 3, 4, 5, 6))
    return layer


def test_multilayer_bridge_dispatch_modes_and_fallbacks():
    """Test L0->L1->L2 dispatch across modes 0, 3 (B=64 & B=128), 4, 5, EXTEND, and sp_active."""
    from sglang_tpu.models.muse_spark import MuseSparkTextModel

    H, H_PRE, Q_COL, KV_COL = 512, 256, 256, 128
    QKV_COL = Q_COL + 2 * KV_COL

    calls: list[str] = []

    def _fake_b1b(st, ga, wo, ws, a, b, pfg, wp, wps, wr, peg, *_s):
        calls.append(f"B1b:{st.shape[0]}")
        return st.clone(), torch.zeros(st.shape[0], 256), torch.zeros(st.shape[0], H_PRE, dtype=torch.bfloat16)

    def _fake_b1c(st, ao, a, b, pfg, wp, wps, wr, peg, *_s):
        calls.append(f"B1c:{st.shape[0]}")
        return st.clone(), torch.zeros(st.shape[0], 256), torch.zeros(st.shape[0], H_PRE, dtype=torch.bfloat16)

    def _fake_b2b(st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg, *_s):
        calls.append(f"B2b:{st.shape[0]}:{mo.shape[0]}")
        return st.clone(), torch.zeros(st.shape[0], QKV_COL, dtype=torch.bfloat16), torch.zeros(st.shape[0], Q_COL, dtype=torch.bfloat16)

    def _fake_b2a(st, mo, wp, sp, bg, a, b, ng, *_s):
        calls.append(f"B2a:{st.shape[0]}:{mo.shape[0]}")
        return st.clone(), torch.zeros(st.shape[0], H, dtype=torch.bfloat16)

    def _fake_b2ro(st, mo, wp, sp, bg, a, b, *_s):
        calls.append(f"B2ro:{st.shape[0]}:{mo.shape[0]}")
        return st.clone()

    def _fake_dual(ai, wq, sq, wg, sg):
        calls.append(f"L0_dual:{ai.shape[0]}")
        return torch.zeros(ai.shape[0], QKV_COL, dtype=torch.bfloat16), torch.zeros(ai.shape[0], Q_COL, dtype=torch.bfloat16)

    def _fake_fp8_lin(x, w, s):
        return (torch.matmul(x, w.to(x.dtype)).float() * s).to(x.dtype)

    def _run_3layer(
        mode: int,
        B: int,
        *,
        fb_mode: str = "DECODE",
        sp_active: bool = False,
        sharded_moe: bool = False,
    ):
        calls.clear()
        model = object.__new__(MuseSparkTextModel)
        torch.nn.Module.__init__(model)
        model.config = types.SimpleNamespace(hidden_size=H, rms_norm_eps=1e-5)
        model.pp_group = types.SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.layers_to_capture = []
        model._capture_slots_by_layer = {}
        model.prefill_kernels = None
        model.rotary_emb = types.SimpleNamespace(cos_sin_pairs=None)
        model.start_layer = 0
        model.end_layer = 3
        model.layers = torch.nn.ModuleList([_make_dummy_layer(mode, H=H, H_PRE=H_PRE) for _ in range(3)])
        if sharded_moe:
            for lyr in model.layers:
                lyr.mlp._fused_decode_op = lambda *_a: torch.zeros(B, H_PRE, dtype=torch.bfloat16)
                lyr.mlp._fused_decode_sharded_op = lambda *_a: torch.zeros(B // 8, H_PRE, dtype=torch.bfloat16)
                lyr.mlp._fused_decode_batches = (64, 128)
                lyr.mlp._fused_decode_rank = torch.zeros(1, dtype=torch.int32)
                lyr.mlp._fused_decode_norm = torch.ones(H_PRE, dtype=torch.float32)
                lyr.mlp.experts.w13_weight = torch.zeros(1)
                lyr.mlp.experts.w13_weight_scale = torch.zeros(1)
                lyr.mlp.experts.w2_weight = torch.zeros(1)
                lyr.mlp.experts.w2_weight_scale = torch.zeros(1)
        model.norm = _make_norm(H)
        model._decode_norm_layout = 0
        model._decode_col_bridges = mode

        embeds = torch.randn(B, H, dtype=torch.bfloat16)
        pos = torch.arange(B, dtype=torch.int32)
        fb = _make_fake_fb(fb_mode)

        env_patch = {"WORLD_SIZE": "8"}
        if sharded_moe:
            env_patch["SGLANG_TPU_MUSE_KERNEL_DECODE_RETURN_SHARDED"] = "1"
        with (
            mock.patch.dict(os.environ, env_patch),
            mock.patch(
                "sglang.srt.runtime_context.get_forward",
                return_value=types.SimpleNamespace(sp_active=sp_active),
            ),
            mock.patch(
                "sglang_tpu.fused_rs_sequence_parallel.get_fused_moe_sequence_parallel_plan",
                return_value=(False, False),
            ),
            mock.patch("sglang_tpu.kernels.decode_collective_bridges.get_attn_to_moe_bridge_op", return_value=_fake_b1b),
            mock.patch("sglang_tpu.kernels.decode_collective_bridges.get_post_attn_to_moe_bridge_op", return_value=_fake_b1c),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges.get_moe_to_attn_bridge_op",
                side_effect=lambda tp, *, sharded_moe_in=False: (
                    lambda *a: (calls.append(f"SH={int(sharded_moe_in)}"), _fake_b2b(*a))[1]
                ),
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges.get_moe_post_ag_res_norm_op",
                side_effect=lambda tp, *, sharded_moe_in=False: _fake_b2a,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges.get_moe_post_ag_res_only_op",
                side_effect=lambda tp, *, sharded_moe_in=False: (
                    lambda *a: (calls.append(f"SH={int(sharded_moe_in)}"), _fake_b2ro(*a))[1]
                ),
            ),
            mock.patch("sglang_tpu.kernels.decode_collective_bridges.get_fp8_dual_qkv_gate_op", return_value=_fake_dual),
            mock.patch("sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm.get_muse_decode_fp8_linear_bf16_op", return_value=_fake_fp8_lin),
            mock.patch("sglang_tpu.models.muse_spark.tensor_model_parallel_all_reduce", side_effect=lambda t: t),
        ):
            out = model(None, pos, fb, input_embeds=embeds)
        assert out.shape == (B, H)
        return list(calls)

    # Mode 0: zero bridge calls
    assert _run_3layer(0, 64) == []
    # Mode 3 @ B=64 & B=128 (replicated MoE out [64, H_PRE]): SH=0!
    assert _run_3layer(3, 64) == ["L0_dual:64", "B1b:64", "SH=0", "B2b:64:64", "B1b:64", "SH=0", "B2b:64:64", "B1b:64", "SH=0", "B2ro:64:64"]
    assert _run_3layer(3, 128) == ["L0_dual:128", "B1b:128", "SH=0", "B2b:128:128", "B1b:128", "SH=0", "B2b:128:128", "B1b:128", "SH=0", "B2ro:128:128"]
    # Mode 3 @ B=64 with token-sharded MoE out ([8, H_PRE]): automatically selects SH=1!
    assert _run_3layer(3, 64, sharded_moe=True) == ["L0_dual:64", "B1b:64", "SH=1", "B2b:64:8", "B1b:64", "SH=1", "B2b:64:8", "B1b:64", "SH=1", "B2ro:64:8"]
    # Mode 5 @ B=64: Hybrid B1c + B2b!
    assert _run_3layer(5, 64) == ["L0_dual:64", "B1c:64", "SH=0", "B2b:64:64", "B1c:64", "SH=0", "B2b:64:64", "B1c:64", "SH=0", "B2ro:64:64"]
    # Mode 4 @ B=64: B1b + B2a (keeps BF16 qkv/gate untouched for LOGO ablation!)
    assert _run_3layer(4, 64) == ["B1b:64", "B2a:64:64", "B1b:64", "B2a:64:64", "B1b:64", "SH=0", "B2ro:64:64"]
    # Safe fallbacks: EXTEND and sp_active=True disable bridges completely!
    assert _run_3layer(3, 64, fb_mode="EXTEND") == []
    assert _run_3layer(3, 64, sp_active=True) == []


def test_mosaic_tpu7x_16dev_export_all_7_bridge_kernels():
    """Export all 7 Wave-7 collective-bridge Pallas kernels to StableHLO for TP=8 & TP=16."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import export as jexp
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.kernels.decode_collective_bridges import (
        attn_oproj_ar_res_norm_jax,
        attn_to_moe_bridge_jax,
        fp8_dual_qkv_gate_jax,
        moe_post_ag_res_norm_jax,
        moe_post_ag_res_only_jax,
        moe_to_attn_bridge_jax,
        post_attn_to_moe_bridge_jax,
    )

    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    devs = jax.devices("cpu")
    if len(devs) < 16:
        pytest.skip("Requires XLA_FLAGS='--xla_force_host_platform_device_count=16'")

    sd = jax.ShapeDtypeStruct
    H, HP, E = 8192, 4096, 256
    exported = 0
    for tp in (8, 16):
        mesh = Mesh(np.array(devs[:tp]), ("model",))
        qkv_c, g_c = (1280 * 8, 1024 * 8) if tp == 8 else (768 * 16, 512 * 16)
        for B in (8, 32, 64, 128):
            specs = [
                (
                    lambda st, ga, wo, ws, a, b, pg: attn_oproj_ar_res_norm_jax(
                        st, ga, wo, ws[0], a, b, pg, world_size=tp
                    ),
                    (P(), P(None, "model"), P("model", None), P("model"), P(), P(), P()),
                    (P(), P(), P()),
                    (
                        sd((B, H), jnp.float32),
                        sd((B, H), jnp.bfloat16),
                        sd((H, H), jnp.float8_e4m3fn),
                        sd((tp, 1, H), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                    ),
                ),
                (
                    lambda st, ga, wo, ws, a, b, pg, wp, ps, wr, eg: attn_to_moe_bridge_jax(
                        st, ga, wo, ws[0], a, b, pg, wp, ps, wr, eg, world_size=tp
                    ),
                    (P(), P(None, "model"), P("model", None), P("model"), P(), P(), P(), P(None, "model"), P(None, "model"), P(), P()),
                    (P(), P(), P()),
                    (
                        sd((B, H), jnp.float32),
                        sd((B, H), jnp.bfloat16),
                        sd((H, H), jnp.float8_e4m3fn),
                        sd((tp, 1, H), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H, HP), jnp.float8_e4m3fn),
                        sd((1, HP), jnp.float32),
                        sd((H, E), jnp.bfloat16),
                        sd((HP,), jnp.float32),
                    ),
                ),
                (
                    lambda st, ao, a, b, pg, wp, ps, wr, eg: post_attn_to_moe_bridge_jax(
                        st, ao, a, b, pg, wp, ps, wr, eg, world_size=tp
                    ),
                    (P(), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(), P()),
                    (P(), P(), P()),
                    (
                        sd((B, H), jnp.float32),
                        sd((B, H), jnp.bfloat16),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H, HP), jnp.float8_e4m3fn),
                        sd((1, HP), jnp.float32),
                        sd((H, E), jnp.bfloat16),
                        sd((HP,), jnp.float32),
                    ),
                ),
                (
                    lambda st, mo, wp, sp, bg, a, b, ng: moe_post_ag_res_norm_jax(
                        st, mo, wp, sp, bg, a, b, ng, world_size=tp
                    ),
                    (P(), P(), P(None, "model"), P(None, "model"), P(), P(), P(), P()),
                    (P(), P()),
                    (
                        sd((B, H), jnp.float32),
                        sd((B, HP), jnp.bfloat16),
                        sd((HP, H), jnp.float8_e4m3fn),
                        sd((1, H), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                    ),
                ),
                (
                    lambda st, mo, wp, sp, bg, a, b: moe_post_ag_res_only_jax(
                        st, mo, wp, sp, bg, a, b, world_size=tp
                    ),
                    (P(), P(), P(None, "model"), P(None, "model"), P(), P(), P()),
                    P(),
                    (
                        sd((B, H), jnp.float32),
                        sd((B, HP), jnp.bfloat16),
                        sd((HP, H), jnp.float8_e4m3fn),
                        sd((1, H), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                    ),
                ),
                (
                    lambda st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg: moe_to_attn_bridge_jax(
                        st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg, world_size=tp
                    ),
                    (P(), P(), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model")),
                    (P(), P(None, "model"), P(None, "model")),
                    (
                        sd((B, H), jnp.float32),
                        sd((B, HP), jnp.bfloat16),
                        sd((HP, H), jnp.float8_e4m3fn),
                        sd((1, H), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H,), jnp.float32),
                        sd((H, qkv_c), jnp.float8_e4m3fn),
                        sd((1, qkv_c), jnp.float32),
                        sd((H, g_c), jnp.float8_e4m3fn),
                        sd((1, g_c), jnp.float32),
                    ),
                ),
                (
                    lambda ai, wq, sq, wg, sg: fp8_dual_qkv_gate_jax(ai, wq, sq, wg, sg),
                    (P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model")),
                    (P(None, "model"), P(None, "model")),
                    (
                        sd((B, H), jnp.bfloat16),
                        sd((H, qkv_c), jnp.float8_e4m3fn),
                        sd((1, qkv_c), jnp.float32),
                        sd((H, g_c), jnp.float8_e4m3fn),
                        sd((1, g_c), jnp.float32),
                    ),
                ),
            ]
            for fn, ins, outs, args in specs:
                with tpu_info_for_cpu_export():
                    jexp.export(
                        jax.jit(jax.shard_map(fn, mesh=mesh, in_specs=ins, out_specs=outs, check_vma=False)),
                        platforms=("tpu",),
                    )(*args)
                exported += 1
            if B >= 8 * tp and (B // tp) % 8 == 0:
                for sh_fn, sh_ins, sh_outs, sh_args in [
                    (
                        lambda st, mo, wp, sp, bg, a, b, ng: moe_post_ag_res_norm_jax(
                            st, mo, wp, sp, bg, a, b, ng, world_size=tp, sharded_moe_in=True
                        ),
                        (P(), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P()),
                        (P(), P()),
                        (sd((B, H), jnp.float32), sd((B, HP), jnp.bfloat16), sd((HP, H), jnp.float8_e4m3fn), sd((1, H), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32)),
                    ),
                    (
                        lambda st, mo, wp, sp, bg, a, b: moe_post_ag_res_only_jax(
                            st, mo, wp, sp, bg, a, b, world_size=tp, sharded_moe_in=True
                        ),
                        (P(), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P()),
                        P(),
                        (sd((B, H), jnp.float32), sd((B, HP), jnp.bfloat16), sd((HP, H), jnp.float8_e4m3fn), sd((1, H), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32)),
                    ),
                    (
                        lambda st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg: moe_to_attn_bridge_jax(
                            st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg, world_size=tp, sharded_moe_in=True
                        ),
                        (P(), P("model", None), P(None, "model"), P(None, "model"), P(), P(), P(), P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model")),
                        (P(), P(None, "model"), P(None, "model")),
                        (sd((B, H), jnp.float32), sd((B, HP), jnp.bfloat16), sd((HP, H), jnp.float8_e4m3fn), sd((1, H), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32), sd((H,), jnp.float32), sd((H, qkv_c), jnp.float8_e4m3fn), sd((1, qkv_c), jnp.float32), sd((H, g_c), jnp.float8_e4m3fn), sd((1, g_c), jnp.float32)),
                    ),
                ]:
                    with tpu_info_for_cpu_export():
                        jexp.export(
                            jax.jit(jax.shard_map(sh_fn, mesh=mesh, in_specs=sh_ins, out_specs=sh_outs, check_vma=False)),
                            platforms=("tpu",),
                        )(*sh_args)
                    exported += 1
    assert exported == 65
