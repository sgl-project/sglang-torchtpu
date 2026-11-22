# SPDX-License-Identifier: Apache-2.0
"""Unit, Pallas interpret parity, and TPU7x Mosaic export tests for Wave-8 carb bridges & LM head."""

from __future__ import annotations

import ast
import inspect
import os
import types
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def test_w8_carb_env_defaults_and_ast_safety():
    """All new Wave-8 bridge & LM-head flags must default off and never invoke EnvField.__bool__."""
    from sglang_tpu.envs import envs

    assert envs.SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT.default is False
    assert envs.SGLANG_TPU_MUSE_DECODE_BRIDGE_MAX_TOKENS.default == 0
    assert envs.SGLANG_TPU_MUSE_DECODE_BRIDGE1_FUSE_TOPK.default is False
    assert envs.SGLANG_TPU_MUSE_DECODE_BRIDGE2_DIRECT_QKV_EPI.default is False
    assert envs.SGLANG_TPU_MUSE_DECODE_BRIDGE2_OVERLAP_EPI.default is False
    assert envs.SGLANG_TPU_MUSE_GREEDY_TOPK1_AUTO.default is False
    assert envs.SGLANG_TPU_MUSE_FAST_FUSED_LM_HEAD_TOPK.default is False
    assert envs.SGLANG_TPU_MUSE_FUSED_GUMBEL_MAX_LM_HEAD.default is False

    flag_names = (
        "SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT",
        "SGLANG_TPU_MUSE_DECODE_BRIDGE_MAX_TOKENS",
        "SGLANG_TPU_MUSE_DECODE_BRIDGE1_FUSE_TOPK",
        "SGLANG_TPU_MUSE_DECODE_BRIDGE2_DIRECT_QKV_EPI",
        "SGLANG_TPU_MUSE_DECODE_BRIDGE2_OVERLAP_EPI",
        "SGLANG_TPU_MUSE_GREEDY_TOPK1_AUTO",
        "SGLANG_TPU_MUSE_FAST_FUSED_LM_HEAD_TOPK",
        "SGLANG_TPU_MUSE_FUSED_GUMBEL_MAX_LM_HEAD",
    )
    for rel in (
        "sglang_tpu/decode_qkv_epilogue_tpu.py",
        "sglang_tpu/distributed_topk_pallas.py",
        "sglang_tpu/sampler.py",
        "sglang_tpu/compilation/decoder_compile_policy.py",
        "sglang_tpu/kernels/decode_collective_bridges.py",
        "sglang_tpu/kernels/decode_collective_bridges_carb.py",
        "sglang_tpu/kernels/fp8_lm_head_fused.py",
        "sglang_tpu/models/muse_spark.py",
    ):
        tree = ast.parse((_REPO_ROOT / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While)):
                src = ast.unparse(node.test)
                for fn in flag_names:
                    if f"envs.{fn}" in src or f"tpu_envs.{fn}" in src:
                        assert ".get()" in src, f"{rel}: bare EnvField bool in `{src}`"


def test_carb_bridge_in_specs_and_prewarm_lock_free():
    """Verify static-arg None padding and lock-free prewarming for all carb bridge ops."""
    from sglang_tpu.kernels import decode_collective_bridges_carb as dcbc

    captured = {}

    def _fake_export(
        name,
        fn,
        *,
        mesh=None,
        input_partition_specs=None,
        output_partition_specs=None,
    ):
        sig = inspect.signature(fn)
        captured[name] = (len(sig.parameters), input_partition_specs, output_partition_specs)
        return fn

    class _ExplodingLock:
        def __enter__(self):
            raise AssertionError("_CARB_LOCK entered after prewarm!")

        def __exit__(self, *args):
            return False

    fake_mesh = object()
    with (
        mock.patch.object(dcbc, "_get_tp_mesh", return_value=fake_mesh),
        mock.patch("sglang_tpu.pallas_jax_op.cpu_export_jax_op", side_effect=_fake_export),
        mock.patch.dict(dcbc._ATTN_TO_MOE_RTOPK_OPS, {}, clear=True),
        mock.patch.dict(dcbc._MOE_TO_ATTN_OVERLAP_OPS, {}, clear=True),
        mock.patch.dict(dcbc._MOE_TO_ATTN_DIRECT_QKV_OPS, {}, clear=True),
        mock.patch.dict(dcbc._DUAL_DIRECT_QKV_OPS, {}, clear=True),
    ):
        dcbc.prewarm_carb_bridge_ops(
            8,
            bf16_router=True,
            fuse_b1_topk=True,
            fuse_b2_direct_qkv=True,
            overlap_b2_epi=True,
        )
        with mock.patch.object(dcbc, "_CARB_LOCK", _ExplodingLock()):
            assert callable(dcbc.get_attn_to_moe_bridge_router_topk_op(8, bf16_router=True))
            assert callable(dcbc.get_moe_to_attn_overlap_bridge_op(8, sharded_moe_in=False))
            assert callable(dcbc.get_moe_to_attn_overlap_bridge_op(8, sharded_moe_in=True))
            assert callable(dcbc.get_fp8_dual_direct_qkv_epi_gate_op(has_rope=True))
            assert callable(dcbc.get_fp8_dual_direct_qkv_epi_gate_op(has_rope=False))
            for hr in (False, True):
                for sh in (False, True):
                    assert callable(
                        dcbc.get_moe_to_attn_direct_qkv_epi_bridge_op(
                            8, has_rope=hr, sharded_moe_in=sh, overlap_mate_epi=True
                        )
                    )

    for op_name, (n_params, in_specs, _out_specs) in captured.items():
        if in_specs is not None:
            assert len(in_specs) == n_params, f"{op_name}: {len(in_specs)} != {n_params}"


def test_dflash_verify_bridge_lift_b320_b384_and_multilayer_relay():
    """Verify VERIFY_BRIDGE_LIFT=1 unlocks B=320 (dq=5) & B=384 (dq=6) + direct QKV relay."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_tpu.decode_qkv_epilogue_tpu import can_use_muse_decode_qkv_epilogue
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.muse_spark import (
        MuseSparkTextModel,
        _decode_bridge_max_tokens,
        _decode_max_tokens,
    )
    from test.registered.unit.hardware_backend.tpu.test_muse_w7_col_proj import (
        _make_dummy_layer,
        _make_fake_fb,
        _make_norm,
    )

    # Default off: caps remain 256 / min(256, 320) (integrate default bridge cap).
    assert _decode_max_tokens() == 256
    assert _decode_bridge_max_tokens() == 256

    qkv_320 = torch.zeros((320, 1280), dtype=torch.bfloat16)
    rope = SimpleNamespace(cos_sin_pairs=torch.zeros((64, 128)))
    backend = SimpleNamespace(
        can_forward_decode_prepared=lambda *_: True,
        forward_decode_prepared_gated=lambda q, kv, g, *_: g.clone(),
    )
    fb_tv = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        attn_backend=backend,
        return_logprob=False,
        capture_hidden_mode=None,
        token_indices_to_pool=None,
        multi_item_delimiter_indices=None,
        extend_seq_lens=None,
    )
    with tpu_envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.override(True):
        # Without VERIFY_BRIDGE_LIFT or DECODE_B384, B=320 is > 256.
        assert not can_use_muse_decode_qkv_epilogue(qkv_320, rope, object(), fb_tv)
        with tpu_envs.SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT.override(True):
            assert _decode_max_tokens() == 384
            assert _decode_bridge_max_tokens() == 384
            assert can_use_muse_decode_qkv_epilogue(qkv_320, rope, object(), fb_tv)

    # Test 3-layer TARGET_VERIFY @ B=320 & B=384 with BRIDGE1_FUSE_TOPK + BRIDGE2_DIRECT_QKV_EPI!
    H, H_PRE = 512, 256
    calls: list[str] = []

    def _fake_rtopk(st, ga, wo, ws, a, b, pfg, wp, wps, wr, rb, peg, *_s):
        calls.append(f"B1_rtopk:{st.shape[0]}:{tuple(rb.shape)}")
        packed = torch.zeros((st.shape[0], 256), dtype=torch.float32)
        packed[:, :8] = 0.125
        packed[:, 8:16] = torch.arange(8, dtype=torch.float32)
        packed[0, 128:136] = float(st.shape[0])
        return st.clone(), packed, torch.zeros((st.shape[0], H_PRE), dtype=torch.bfloat16)

    def _fake_b2_dqkv(st, mo, *_rest):
        B = st.shape[0]
        calls.append(f"B2_dqkv:{B}:{mo.shape[0]}")
        return (
            st.clone(),
            torch.zeros((1, B, 8, 2, 128), dtype=torch.bfloat16),
            torch.zeros((B, 1, 4, 128), dtype=torch.float8_e4m3fn),
            torch.zeros((B, 1024), dtype=torch.bfloat16),
        )

    def _fake_l0_dqkv(ai, *_rest):
        B = ai.shape[0]
        calls.append(f"L0_dqkv:{B}")
        return (
            torch.zeros((1, B, 8, 2, 128), dtype=torch.bfloat16),
            torch.zeros((B, 1, 4, 128), dtype=torch.float8_e4m3fn),
            torch.zeros((B, 1024), dtype=torch.bfloat16),
        )

    def _fake_b2ro(st, mo, *_rest):
        calls.append(f"B2_ro:{st.shape[0]}:{mo.shape[0]}")
        return st.clone()

    for B in (320, 384):
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
        layers = []
        for idx in range(3):
            lyr = _make_dummy_layer(3, H=H, H_PRE=H_PRE)
            lyr._decode_fuse_target_verify = True
            lyr.self_attn.q_size = 1024
            lyr.self_attn.kv_size = 128
            lyr.self_attn.decode_qkv_epilogue = True
            lyr.self_attn.decode_attn_gate_fuse = True
            lyr.self_attn.qkv_proj.weight = torch.nn.Parameter(
                torch.randn(1280, H, dtype=torch.bfloat16) * 0.02, requires_grad=False
            )
            lyr.self_attn.gate_proj.weight = torch.nn.Parameter(
                torch.randn(1024, H, dtype=torch.bfloat16) * 0.02, requires_grad=False
            )
            lyr.self_attn.o_proj.weight = torch.nn.Parameter(
                torch.randn(H, 1024, dtype=torch.bfloat16) * 0.02, requires_grad=False
            )
            lyr.self_attn._o_proj_w8k = None
            assert lyr.self_attn.prepare_w7_qkv_gate_fp8()
            assert lyr.self_attn.prepare_w7_o_proj_fp8()
            layers.append(lyr)
        model.layers = torch.nn.ModuleList(layers)
        model.norm = _make_norm(H)
        model._decode_norm_layout = 0
        model._decode_col_bridges = 3

        embeds = torch.randn(B, H, dtype=torch.bfloat16)
        pos = torch.arange(B, dtype=torch.int32)

        with (
            tpu_envs.SGLANG_TPU_MUSE_VERIFY_BRIDGE_LIFT.override(True),
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST.override(True),
            tpu_envs.SGLANG_TPU_MUSE_DECODE_BRIDGE1_FUSE_TOPK.override(True),
            tpu_envs.SGLANG_TPU_MUSE_DECODE_BRIDGE2_DIRECT_QKV_EPI.override(True),
            tpu_envs.SGLANG_TPU_MUSE_DECODE_BRIDGE2_OVERLAP_EPI.override(True),
            mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}),
            mock.patch(
                "sglang.srt.runtime_context.get_forward",
                return_value=types.SimpleNamespace(sp_active=False),
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_carb.get_attn_to_moe_bridge_router_topk_op",
                return_value=_fake_rtopk,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_carb.get_moe_to_attn_direct_qkv_epi_bridge_op",
                return_value=_fake_b2_dqkv,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges_carb.get_fp8_dual_direct_qkv_epi_gate_op",
                return_value=_fake_l0_dqkv,
            ),
            mock.patch(
                "sglang_tpu.kernels.decode_collective_bridges.get_moe_post_ag_res_only_op",
                return_value=_fake_b2ro,
            ),
        ):
            out = model(None, pos, fb_tv, input_embeds=embeds)
        assert out.shape == (B, H)
        assert calls == [
            f"L0_dqkv:{B}",
            f"B1_rtopk:{B}:(1, 256)",
            f"B2_dqkv:{B}:{B}",
            f"B1_rtopk:{B}:(1, 256)",
            f"B2_dqkv:{B}:{B}",
            f"B1_rtopk:{B}:(1, 256)",
            f"B2_ro:{B}:{B}",
        ]


def test_pallas_cpu_interpret_router_topk_and_direct_qkv_parity():
    """Verify exact numerical parity via Pallas CPU interpret mode for new carb kernels."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.attention.pallas.decode_qkv_epilogue import (
        build_decode_rope_step_tables,
        run_decode_qkv_epilogue,
    )
    from sglang_tpu.kernels.decode_collective_bridges_carb import (
        _pack_router_topk_and_counts_in_vmem,
        fp8_dual_direct_qkv_epi_gate_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_routing import (
        _route_prerouted_dense_in_kernel,
        _route_prerouted_topk_in_kernel,
        dense_routing_scratch_shapes,
        route_topk_and_counts_in_kernel,
        route_weights,
        routing_scratch_shapes,
    )

    rng = np.random.default_rng(42)
    B, E = 32, 256
    logits_np = rng.standard_normal((B, E)).astype(np.float32) * 1.5
    logits_np[31, :] = np.nan  # Padded NaN row safety check!
    bias_np = (rng.standard_normal((1, E)) * 0.1).astype(np.float32)

    # 1. Pack via _pack_router_topk_and_counts_in_vmem and compare vs reference route_weights
    #    and route_topk_and_counts_in_kernel!
    v_sp = pl.BlockSpec(memory_space=pltpu.VMEM)
    packed = pl.pallas_call(
        lambda lg_ref, b_ref, out_ref: _pack_router_topk_and_counts_in_vmem(
            lg_ref, b_ref, out_ref, rows=B, top_k=8, route_eps=1e-15, route_scale=1.25
        ),
        out_shape=jax.ShapeDtypeStruct((B, E), jnp.float32),
        in_specs=[v_sp, v_sp],
        out_specs=v_sp,
        interpret=pltpu.InterpretParams(detect_races=True),
    )(jnp.asarray(logits_np), jnp.asarray(bias_np))

    ref_dense = route_weights(
        jnp.asarray(logits_np), jnp.asarray(bias_np), topk=8, route_eps=1e-15, route_scale=1.25
    )

    # Unpack via _route_prerouted_dense_in_kernel and compare dense_w + counts!
    d_shapes = dense_routing_scratch_shapes(B, 32, num_experts=E)

    def _run_dense_unpack(pk_ref, out_w_ref, out_c_ref, d_sc, sem):
        _route_prerouted_dense_in_kernel(pk_ref, d_sc, sem, topk=8, route_chunk_rows=16)
        out_w_ref[...] = d_sc.dense_w_vmem[...]
        out_c_ref[...] = d_sc.counts_vmem[...]

    got_dense, got_counts = pl.pallas_call(
        _run_dense_unpack,
        out_shape=(
            jax.ShapeDtypeStruct((B, E), jnp.float32),
            jax.ShapeDtypeStruct((8, E), jnp.int32),
        ),
        in_specs=[v_sp],
        out_specs=(v_sp, v_sp),
        scratch_shapes=[d_shapes, pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(detect_races=False),
    )(packed)

    np.testing.assert_allclose(np.asarray(got_dense), np.asarray(ref_dense), rtol=1e-6, atol=1e-6)

    # Compare sparse hit compaction (_route_prerouted_topk_in_kernel vs route_topk_and_counts_in_kernel)
    r_shapes = routing_scratch_shapes(B, 34, topk=8, num_experts=E)
    scaled_packed = packed.at[:, :8].set(packed[:, :8] / 1.25)

    def _run_ref_sparse(lg_ref, b_ref, op_ref, oc_ref, r_sc, sem):
        route_topk_and_counts_in_kernel(
            lg_ref, b_ref, r_sc, sem,
            my_owner=jnp.int32(2), e_per_owner=32,
            holder_cand_eids=(jnp.int32(5), jnp.int32(200)),
            topk=8, route_eps=1e-15, route_scale=1.0, route_chunk_rows=16,
        )
        op_ref[...] = r_sc.route_pack_vmem[...]
        oc_ref[...] = r_sc.counts_vmem[...]

    def _run_pre_sparse(pk_ref, op_ref, oc_ref, r_sc, sem):
        _route_prerouted_topk_in_kernel(
            pk_ref, r_sc, sem,
            my_owner=jnp.int32(2), e_per_owner=32,
            holder_cand_eids=(jnp.int32(5), jnp.int32(200)),
            topk=8, route_chunk_rows=16,
        )
        op_ref[...] = r_sc.route_pack_vmem[...]
        oc_ref[...] = r_sc.counts_vmem[...]

    ref_pack, ref_cnt = pl.pallas_call(
        _run_ref_sparse,
        out_shape=(
            jax.ShapeDtypeStruct((B, 128), jnp.int32),
            jax.ShapeDtypeStruct((8, E), jnp.int32),
        ),
        in_specs=[v_sp, v_sp],
        out_specs=(v_sp, v_sp),
        scratch_shapes=[r_shapes, pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(detect_races=False),
    )(jnp.asarray(logits_np), jnp.asarray(bias_np))

    got_pack, got_cnt = pl.pallas_call(
        _run_pre_sparse,
        out_shape=(
            jax.ShapeDtypeStruct((B, 128), jnp.int32),
            jax.ShapeDtypeStruct((8, E), jnp.int32),
        ),
        in_specs=[v_sp],
        out_specs=(v_sp, v_sp),
        scratch_shapes=[r_shapes, pltpu.SemaphoreType.DMA],
        interpret=pltpu.InterpretParams(detect_races=False),
    )(scaled_packed)

    np.testing.assert_array_equal(np.asarray(got_cnt), np.asarray(ref_cnt))
    np.testing.assert_array_equal(np.asarray(got_pack[:, :16]), np.asarray(ref_pack[:, :16]))
    ref_w_bits = jax.lax.bitcast_convert_type(ref_pack[:, 16:24], jnp.float32)
    got_w_bits = jax.lax.bitcast_convert_type(got_pack[:, 16:24], jnp.float32)
    np.testing.assert_allclose(np.asarray(got_w_bits), np.asarray(ref_w_bits), rtol=1e-6, atol=1e-6)

    # 2. Direct L0 QKV+RoPE epilogue parity vs fp8_dual_qkv_gate + run_decode_qkv_epilogue!
    H = 256
    attn_in = jnp.asarray(rng.standard_normal((64, H)), dtype=jnp.bfloat16)
    wq = jnp.asarray(rng.standard_normal((H, 1280)) * 0.1, dtype=jnp.float8_e4m3fn)
    sq = jnp.asarray(np.abs(rng.standard_normal((1, 1280))) * 0.05 + 0.01, dtype=jnp.float32)
    wg = jnp.asarray(rng.standard_normal((H, 1024)) * 0.1, dtype=jnp.float8_e4m3fn)
    sg = jnp.asarray(np.abs(rng.standard_normal((1, 1024))) * 0.05 + 0.01, dtype=jnp.float32)
    pos = jnp.arange(64, dtype=jnp.int32)
    cs_pairs = jnp.asarray(rng.standard_normal((128, 128)), dtype=jnp.float32)
    cs_step = build_decode_rope_step_tables(pos, cs_pairs)

    for has_rope in (False, True):
        cs_arg = cs_step if has_rope else None
        q_got, kv_got, gate_got = fp8_dual_direct_qkv_epi_gate_jax(
            attn_in, wq, sq, wg, sg, cs_arg,
            qkv_norm_eps=1e-6, query_scale=0.125, k_scale=0.5, v_scale=0.25,
            interpret=True,
        )
        qkv_ref = (jnp.dot(attn_in, wq, preferred_element_type=jnp.float32) * sq).astype(jnp.bfloat16)
        gate_ref = (jnp.dot(attn_in, wg, preferred_element_type=jnp.float32) * sg).astype(jnp.bfloat16)
        q_3d_ref, kv_ref = run_decode_qkv_epilogue(
            qkv_ref, cs_arg,
            norm_eps=1e-6, query_scale=0.125, k_scale=0.5, v_scale=0.25,
            interpret=True,
        )
        np.testing.assert_array_equal(
            np.asarray(q_got.view(jnp.uint16)),
            np.asarray(q_3d_ref.reshape(1, 64, 8, 2, 128).view(jnp.uint16)),
        )
        np.testing.assert_array_equal(
            np.asarray(jax.lax.bitcast_convert_type(kv_got, jnp.uint8)),
            np.asarray(jax.lax.bitcast_convert_type(kv_ref, jnp.uint8)),
        )
        np.testing.assert_array_equal(
            np.asarray(gate_got.view(jnp.uint16)),
            np.asarray(gate_ref.view(jnp.uint16)),
        )


def test_gumbel_max_and_greedy_topk1_auto_dispatch_and_distribution():
    """Verify Gumbel-max statistical accuracy, T=0 greedy fallback, and sampler/policy dispatch."""
    import jax.numpy as jnp
    from jax import lax

    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.model_executor.forward_batch_info import ForwardMode, LogitsOutputMode
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.sampler import TpuSampler, select_logits_output_mode
    from test.registered.unit.hardware_backend.tpu.test_muse_fp8_lm_head import _make_fake_lm_head

    # 1. Verify SplitMix32 Gumbel-max categorical distribution matches exact softmax(z / T)!
    logits = jnp.array([[1.4, -0.3, 0.8, 2.1, -1.0, 0.2]], dtype=jnp.float32)
    temp = 0.7
    probs_ref = np.exp((np.asarray(logits[0]) / temp))
    probs_ref /= probs_ref.sum()

    n_trials = 8192
    rng = np.random.default_rng(7)
    seeds_f32 = rng.random((n_trials, 6), dtype=np.float32)
    seed_u32 = lax.bitcast_convert_type(jnp.asarray(seeds_f32), jnp.uint32)
    c_q = jnp.arange(6, dtype=jnp.uint32)[None, :]
    row_id = jnp.arange(n_trials, dtype=jnp.uint32)[:, None]
    h = (row_id * jnp.uint32(0xBF58476D)) ^ seed_u32
    h = h + c_q * jnp.uint32(0x94D049BB)
    h = h ^ lax.shift_right_logical(h, jnp.uint32(16))
    h = h * jnp.uint32(0x85EBCA6B)
    h = h ^ lax.shift_right_logical(h, jnp.uint32(13))
    h = h * jnp.uint32(0xC2B2AE35)
    h = h ^ lax.shift_right_logical(h, jnp.uint32(16))
    u = (
        lax.shift_right_logical(h, jnp.uint32(8)).astype(jnp.float32)
        * jnp.float32(1.0 / 16777216.0)
        + jnp.float32(0.5 / 16777216.0)
    )
    gumbel = -jnp.log(-jnp.log(u))
    picks = np.asarray(jnp.argmax(logits / temp + gumbel, axis=-1))
    emp = np.bincount(picks, minlength=6) / float(n_trials)
    np.testing.assert_allclose(emp, probs_ref, atol=0.022)

    # 2. Verify select_logits_output_mode routes GREEDY_TOPK1_AUTO -> DISTRIBUTED_GREEDY
    #    and unconstrained T>0 -> DISTRIBUTED_GUMBEL_MAX!
    head = _make_fake_lm_head()
    lp = SimpleNamespace(
        use_tpu_distributed_top_k=True,
        use_fp32_lm_head=True,
        do_tensor_parallel_all_gather=True,
        do_tensor_parallel_all_gather_dp_attn=False,
        final_logit_softcapping=None,
    )
    model = SimpleNamespace(lm_head=head, logits_processor=lp, config=SimpleNamespace(tie_word_embeddings=False))

    greedy_info = SimpleNamespace(
        is_all_greedy=True,
        is_any_greedy=True,
        uniform_top_k=1,
        sampling_seed=None,
        need_top_k_sampling=False,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        sampling_support_logprobs_capture_indices=None,
        has_custom_logit_processor=False,
        logit_bias=None,
        grammars=None,
        sampling_mask_batch_indices=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=None,
        sampling_params_on_cpu=True,
        temperatures=torch.zeros((32, 1), dtype=torch.float32),
        device="cpu",
    )
    fb_g = SimpleNamespace(
        spec_info=None,
        spec_algorithm=None,
        sampling_info=greedy_info,
        batch_size=64,
        return_logprob=False,
        forward_mode=ForwardMode.DECODE,
        logits_output_mode=LogitsOutputMode.FULL_VOCAB,
    )

    with (
        mock.patch("sglang_tpu.sampler.distributed_top_k_candidate_width_for_model", return_value=64),
        tpu_envs.SGLANG_TPU_DISTRIBUTED_GREEDY_TOPK.override(True),
        tpu_envs.SGLANG_TPU_SPEC_VERIFY_GREEDY_TOPK1.override(False),
    ):
        assert select_logits_output_mode(fb_g, model) == LogitsOutputMode.DISTRIBUTED_TOP_K
        with tpu_envs.SGLANG_TPU_MUSE_GREEDY_TOPK1_AUTO.override(True):
            assert select_logits_output_mode(fb_g, model) == LogitsOutputMode.DISTRIBUTED_GREEDY

    # Unconstrained temperature sampling -> DISTRIBUTED_GUMBEL_MAX + cached gumbel_temperatures!
    temp_info = SimpleNamespace(
        **{
            **greedy_info.__dict__,
            "is_all_greedy": False,
            "is_any_greedy": False,
            "uniform_top_k": None,
            "temperatures": torch.full((48, 1), 0.8, dtype=torch.float32),
        }
    )
    temp_info.__len__ = lambda: 48  # not used on SimpleNamespace; wrap in custom class:

    class _FakeInfo:
        def __init__(self, d):
            self.__dict__.update(d)

        def __len__(self):
            return 48

    fb_t = SimpleNamespace(
        spec_info=None,
        spec_algorithm=None,
        sampling_info=_FakeInfo(temp_info.__dict__),
        batch_size=64,
        return_logprob=False,
        forward_mode=ForwardMode.DECODE,
        logits_output_mode=LogitsOutputMode.FULL_VOCAB,
        gumbel_temperatures=None,
    )
    with (
        mock.patch("sglang_tpu.sampler.distributed_top_k_candidate_width_for_model", return_value=64),
        mock.patch.object(tpu_envs.SGLANG_TPU_MUSE_FUSED_FP8_LM_HEAD_TOPK, "get", return_value=True),
        mock.patch.object(tpu_envs.SGLANG_TPU_MUSE_FUSED_GUMBEL_MAX_LM_HEAD, "get", return_value=True),
    ):
        mode_t = select_logits_output_mode(fb_t, model)
        assert mode_t == LogitsOutputMode.DISTRIBUTED_GUMBEL_MAX
        assert fb_t.gumbel_temperatures is not None
        assert tuple(fb_t.gumbel_temperatures.shape) == (64, 1)
        assert torch.allclose(fb_t.gumbel_temperatures[:48], torch.full((48, 1), 0.8))
        assert torch.allclose(fb_t.gumbel_temperatures[48:], torch.ones((16, 1)))

        # TpuSampler.forward directly consumes 1-col global Gumbel-max token IDs with 0 softmax!
        from sglang_tpu.sampler import _DeviceParamCache

        sampler = TpuSampler.__new__(TpuSampler)
        torch.nn.Module.__init__(sampler)
        sampler._device_params = _DeviceParamCache()
        expected_ids = torch.arange(100, 164, dtype=torch.int32).unsqueeze(1)
        lo = LogitsProcessorOutput(
            next_token_logits=None,
            next_token_global_top_k_logits=torch.ones((64, 1), dtype=torch.float32),
            next_token_global_top_k_token_ids=expected_ids,
        )
        def _fake_replace(obj, **changes):
            obj.__dict__.update(changes)
            return obj

        sampler._sync_token_ids_across_tp = lambda ids, info: None
        with mock.patch("sglang_tpu.sampler.replace", side_effect=_fake_replace), mock.patch.object(
            sampler._device_params, "materialize", side_effect=AssertionError("1-col path must skip params")
        ):
            sampled = sampler(
                lo, fb_t.sampling_info, False, None, None, torch.arange(64, dtype=torch.int32)
            )
        assert sampled.dtype == torch.int64
        assert torch.equal(sampled, expected_ids[:, 0].to(torch.int64))


def test_mosaic_tpu7x_export_b320_b384_carb_bridges_and_lm_head():
    """Export all B=160/320/384 bridges, carb router-topk / direct-QKV bridges, and Gumbel-max LM head."""
    import jax
    import jax.numpy as jnp
    from jax import export as jexp
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.kernels.decode_collective_bridges import (
        attn_to_moe_bridge_jax,
        fp8_dual_qkv_gate_jax,
        moe_post_ag_res_norm_jax,
        moe_post_ag_res_only_jax,
        moe_to_attn_bridge_jax,
    )
    from sglang_tpu.kernels.decode_collective_bridges_carb import (
        attn_to_moe_bridge_router_topk_jax,
        fp8_dual_direct_qkv_epi_gate_jax,
        moe_to_attn_direct_qkv_epi_bridge_jax,
        moe_to_attn_overlap_bridge_jax,
    )
    from sglang_tpu.kernels.fp8_lm_head_fused import (
        fp8_lm_head_fused_topk64_jax,
        fp8_lm_head_gumbel_max_jax,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.attn_ar_res_norm import (
        tp_ar_res_norm_jax,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    devs = jax.devices("cpu")
    if len(devs) < 16:
        pytest.skip("Requires XLA_FLAGS='--xla_force_host_platform_device_count=16'")

    sd = jax.ShapeDtypeStruct
    H, HP, E = 8192, 4096, 256
    tp = 8
    mesh = Mesh(np.array(devs[:tp]), ("model",))
    qkv_c, g_c = 1280 * tp, 1024 * tp
    exported = 0

    with tpu_info_for_cpu_export():
        # 1. Verify B=160 (partial dq=5), B=320 (d=4 dq=5), and B=384 (d=5 dq=6) across r32 & r16!
        for B in (160, 320, 384):
            for r_dt in (jnp.float32, jnp.bfloat16):
                jexp.export(
                    jax.jit(
                        jax.shard_map(
                            lambda st, ga, wo, ws, a, b, pg, wp, ps, wr, eg: attn_to_moe_bridge_jax(
                                st, ga, wo, ws[0], a, b, pg, wp, ps, wr, eg, world_size=tp
                            ),
                            mesh=mesh,
                            in_specs=(
                                P(), P(None, "model"), P("model", None), P("model"),
                                P(), P(), P(), P(None, "model"), P(None, "model"), P(), P(),
                            ),
                            out_specs=(P(), P(), P()),
                            check_vma=False,
                        )
                    ),
                    platforms=("tpu",),
                )(
                    sd((B, H), jnp.float32),
                    sd((B, H), jnp.bfloat16),
                    sd((H, H), jnp.float8_e4m3fn),
                    sd((tp, 1, H), jnp.float32),
                    sd((H,), jnp.float32),
                    sd((H,), jnp.float32),
                    sd((H,), jnp.float32),
                    sd((H, HP), jnp.float8_e4m3fn),
                    sd((1, HP), jnp.float32),
                    sd((H, E), r_dt),
                    sd((HP,), jnp.float32),
                )
                exported += 1

            sh_ok = (B // tp) % 8 == 0
            jexp.export(
                jax.jit(
                    jax.shard_map(
                        lambda st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg: moe_to_attn_bridge_jax(
                            st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg,
                            world_size=tp, sharded_moe_in=sh_ok,
                        ),
                        mesh=mesh,
                        in_specs=(
                            P(), P("model", None) if sh_ok else P(),
                            P(None, "model"), P(None, "model"),
                            P(), P(), P(), P(),
                            P(None, "model"), P(None, "model"),
                            P(None, "model"), P(None, "model"),
                        ),
                        out_specs=(P(), P(None, "model"), P(None, "model")),
                        check_vma=False,
                    )
                ),
                platforms=("tpu",),
            )(
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
            )
            exported += 1

            jexp.export(
                jax.jit(
                    jax.shard_map(
                        lambda st, mo, wp, sp, bg, a, b, ng: moe_post_ag_res_norm_jax(
                            st, mo, wp, sp, bg, a, b, ng, world_size=tp, sharded_moe_in=sh_ok
                        ),
                        mesh=mesh,
                        in_specs=(
                            P(), P("model", None) if sh_ok else P(),
                            P(None, "model"), P(None, "model"), P(), P(), P(), P(),
                        ),
                        out_specs=(P(), P()),
                        check_vma=False,
                    )
                ),
                platforms=("tpu",),
            )(
                sd((B, H), jnp.float32),
                sd((B, HP), jnp.bfloat16),
                sd((HP, H), jnp.float8_e4m3fn),
                sd((1, H), jnp.float32),
                sd((H,), jnp.float32),
                sd((H,), jnp.float32),
                sd((H,), jnp.float32),
                sd((H,), jnp.float32),
            )
            exported += 1

            jexp.export(
                jax.jit(
                    jax.shard_map(
                        lambda st, mo, wp, sp, bg, a, b: moe_post_ag_res_only_jax(
                            st, mo, wp, sp, bg, a, b, world_size=tp, sharded_moe_in=sh_ok
                        ),
                        mesh=mesh,
                        in_specs=(
                            P(), P("model", None) if sh_ok else P(),
                            P(None, "model"), P(None, "model"), P(), P(), P(),
                        ),
                        out_specs=P(),
                        check_vma=False,
                    )
                ),
                platforms=("tpu",),
            )(
                sd((B, H), jnp.float32),
                sd((B, HP), jnp.bfloat16),
                sd((HP, H), jnp.float8_e4m3fn),
                sd((1, H), jnp.float32),
                sd((H,), jnp.float32),
                sd((H,), jnp.float32),
                sd((H,), jnp.float32),
            )
            exported += 1

            jexp.export(
                jax.jit(
                    jax.shard_map(
                        lambda ai, wq, sq, wg, sg: fp8_dual_qkv_gate_jax(ai, wq, sq, wg, sg),
                        mesh=mesh,
                        in_specs=(P(), P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model")),
                        out_specs=(P(None, "model"), P(None, "model")),
                        check_vma=False,
                    )
                ),
                platforms=("tpu",),
            )(
                sd((B, H), jnp.bfloat16),
                sd((H, qkv_c), jnp.float8_e4m3fn),
                sd((1, qkv_c), jnp.float32),
                sd((H, g_c), jnp.float8_e4m3fn),
                sd((1, g_c), jnp.float32),
            )
            exported += 1

            jexp.export(
                jax.jit(
                    jax.shard_map(
                        lambda st, op, a, b, ng: tp_ar_res_norm_jax(
                            st, op, a, b, ng, world_size=tp, branch_eps=1e-8, next_eps=1e-5
                        ),
                        mesh=mesh,
                        in_specs=(P(), P(), P(), P(), P()),
                        out_specs=(P(), P(), P()),
                        check_vma=False,
                    )
                ),
                platforms=("tpu",),
            )(
                sd((B, H), jnp.float32),
                sd((B, H), jnp.bfloat16),
                sd((H,), jnp.float32),
                sd((H,), jnp.float32),
                sd((H,), jnp.float32),
            )
            exported += 1

        # 2. Export new carb Bridge 1b router-TopK + Bridge 2b direct-QKV/overlap + L0 direct-QKV!
        for B in (64, 320, 384):
            jexp.export(
                jax.jit(
                    jax.shard_map(
                        lambda st, ga, wo, ws, a, b, pg, wp, ps, wr, rb, eg: attn_to_moe_bridge_router_topk_jax(
                            st, ga, wo, ws[0], a, b, pg, wp, ps, wr, rb, eg, world_size=tp
                        ),
                        mesh=mesh,
                        in_specs=(
                            P(), P(None, "model"), P("model", None), P("model"),
                            P(), P(), P(), P(None, "model"), P(None, "model"), P(), P(), P(),
                        ),
                        out_specs=(P(), P(), P()),
                        check_vma=False,
                    )
                ),
                platforms=("tpu",),
            )(
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
                sd((1, E), jnp.float32),
                sd((HP,), jnp.float32),
            )
            exported += 1

            jexp.export(
                jax.jit(
                    jax.shard_map(
                        lambda st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg: moe_to_attn_overlap_bridge_jax(
                            st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg,
                            world_size=tp, sharded_moe_in=True,
                        ),
                        mesh=mesh,
                        in_specs=(
                            P(), P("model", None), P(None, "model"), P(None, "model"),
                            P(), P(), P(), P(),
                            P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"),
                        ),
                        out_specs=(P(), P(None, "model"), P(None, "model")),
                        check_vma=False,
                    )
                ),
                platforms=("tpu",),
            )(
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
            )
            exported += 1

            for ovl in (False, True):
                jexp.export(
                    jax.jit(
                        jax.shard_map(
                            lambda st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg, cs: moe_to_attn_direct_qkv_epi_bridge_jax(
                                st, mo, wp, sp, bg, a, b, ng, wq, sq, wg, sg, cs,
                                world_size=tp, sharded_moe_in=True, overlap_mate_epi=ovl,
                            ),
                            mesh=mesh,
                            in_specs=(
                                P(), P("model", None), P(None, "model"), P(None, "model"),
                                P(), P(), P(), P(),
                                P(None, "model"), P(None, "model"), P(None, "model"), P(None, "model"),
                                P(),
                            ),
                            out_specs=(
                                P(),
                                P(None, None, "model", None, None),
                                P(None, "model", None, None),
                                P(None, "model"),
                            ),
                            check_vma=False,
                        )
                    ),
                    platforms=("tpu",),
                )(
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
                    sd((B, 2, 128), jnp.float32),
                )
                exported += 1

            jexp.export(
                jax.jit(
                    jax.shard_map(
                        lambda ai, wq, sq, wg, sg, cs: fp8_dual_direct_qkv_epi_gate_jax(
                            ai, wq, sq, wg, sg, cs
                        ),
                        mesh=mesh,
                        in_specs=(
                            P(), P(None, "model"), P(None, "model"),
                            P(None, "model"), P(None, "model"), P(),
                        ),
                        out_specs=(
                            P(None, None, "model", None, None),
                            P(None, "model", None, None),
                            P(None, "model"),
                        ),
                        check_vma=False,
                    )
                ),
                platforms=("tpu",),
            )(
                sd((B, H), jnp.bfloat16),
                sd((H, qkv_c), jnp.float8_e4m3fn),
                sd((1, qkv_c), jnp.float32),
                sd((H, g_c), jnp.float8_e4m3fn),
                sd((1, g_c), jnp.float32),
                sd((B, 2, 128), jnp.float32),
            )
            exported += 1

        # 3. Export Gumbel-max LM head + XOR-butterfly tn=1024 Top-64 LM head on TP=8 & TP=16!
        for w_tp in (8, 16):
            m_tp = Mesh(np.array(devs[:w_tp]), ("model",))
            v_loc = 202048 // w_tp
            vp = -(-v_loc // 1024) * 1024
            for B in (64, 320, 384):
                exp_g = jexp.export(
                    jax.jit(
                        jax.shard_map(
                            lambda x, w, s, t, seed: fp8_lm_head_gumbel_max_jax(
                                x, w, s, t, seed,
                                world_size=w_tp,
                                local_width=v_loc,
                                org_vocab_size=202048,
                                global_vocab_size=202048,
                                org_vocab_padded_count=v_loc,
                                logit_scale=0.1767767,
                                tn=1024,
                            ),
                            mesh=m_tp,
                            in_specs=(P(), P(None, "model"), P(None, "model"), P(), P()),
                            out_specs=(P(), P()),
                            check_vma=False,
                        )
                    ),
                    platforms=("tpu",),
                )(
                    sd((B, H), jnp.bfloat16),
                    sd((H, vp * w_tp), jnp.float8_e4m3fn),
                    sd((1, vp * w_tp), jnp.float32),
                    sd((B, 1), jnp.float32),
                    sd((1, 128), jnp.float32),
                )
                assert tuple(a.shape for a in exp_g.out_avals) == ((B, 1), (B, 1))
                exported += 1

                exp_tk = jexp.export(
                    jax.jit(
                        jax.shard_map(
                            lambda x, w, s: fp8_lm_head_fused_topk64_jax(
                                x, w, s,
                                world_size=w_tp,
                                local_width=v_loc,
                                org_vocab_size=202048,
                                global_vocab_size=202048,
                                org_vocab_padded_count=v_loc,
                                k=64,
                                logit_scale=0.1767767,
                                tn=1024,
                                exact_tie_break=False,
                                use_xor_butterfly=True,
                            ),
                            mesh=m_tp,
                            in_specs=(P(), P(None, "model"), P(None, "model")),
                            out_specs=(P(), P()),
                            check_vma=False,
                        )
                    ),
                    platforms=("tpu",),
                )(
                    sd((B, H), jnp.bfloat16),
                    sd((H, vp * w_tp), jnp.float8_e4m3fn),
                    sd((1, vp * w_tp), jnp.float32),
                )
                assert tuple(a.shape for a in exp_tk.out_avals) == ((B, 64), (B, 64))
                exported += 1

    assert exported == 48


def test_w7_b1b_b2b_interpret_parity_and_b80_direct_qkv_export():
    """Verify 8-rank interpret=True numerical parity for optimized Bridge 1b/2b and B=80 dqkv."""
    import re
    import jax
    import jax.numpy as jnp
    from jax import export as jexp
    from jax._src import tpu_custom_call
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from sglang_tpu.kernels.decode_collective_bridges import (
        attn_to_moe_bridge_jax,
        moe_to_attn_bridge_jax,
    )
    from sglang_tpu.kernels.decode_collective_bridges_carb import (
        fp8_dual_direct_qkv_epi_gate_jax,
        moe_to_attn_direct_qkv_epi_bridge_jax,
    )

    all_devs = list(jax.devices())
    if len(all_devs) < 8:
        return
    mesh = Mesh(np.array(all_devs[:8]), ("model",))

    def _check_mlir(name, fn, in_specs, out_specs, shapes_dtypes, *, check_t16: bool = True):
        asms = []
        orig = tpu_custom_call._lower_to_custom_call_config

        def _spy(mod, **kw):
            asms.append(mod.operation.get_asm(enable_debug_info=False))
            return orig(mod, **kw)

        tpu_custom_call._lower_to_custom_call_config = _spy
        try:
            avals = [
                jax.ShapeDtypeStruct(sh, dt, sharding=NamedSharding(mesh, sp))
                for (sh, dt), sp in zip(shapes_dtypes, in_specs)
            ]
            w = jax.jit(
                jax.shard_map(
                    fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
                )
            )
            with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
                exp = jexp.export(w, platforms=("tpu",))(*avals)
            mlir = exp.mlir_module()
        finally:
            tpu_custom_call._lower_to_custom_call_config = orig
        assert "stablehlo.partition_id" not in mlir
        if check_t16:
            bad = [
                ln
                for ln in asms[0].splitlines()
                if ("tpu.concatenate" in ln and "vector<1x" in ln)
                or (
                    ("vector.load" in ln or "vector_store" in ln)
                    and "xbf16>" in ln
                    and "memref<8x" not in ln
                    and re.search(r"vector<(?:\d+x)*(?:1|8)x\d+xbf16>", ln)
                )
            ]
            assert not bad, f"{name}: {bad[:2]}"

    H_FULL, HP_FULL, E_FULL = 8192, 4096, 256
    for B in (16, 32, 64, 128, 384):
        _check_mlir(
            f"b1b_{B}",
            lambda *a: attn_to_moe_bridge_jax(*a, world_size=8),
            (P(),) * 11,
            (P(), P(), P()),
            [
                ((B, H_FULL), jnp.float32),
                ((B, 1024), jnp.bfloat16),
                ((1024, H_FULL), jnp.float8_e4m3fn),
                ((H_FULL,), jnp.float32),
                ((H_FULL,), jnp.float32),
                ((H_FULL,), jnp.float32),
                ((H_FULL,), jnp.float32),
                ((H_FULL, 512), jnp.float8_e4m3fn),
                ((512,), jnp.float32),
                ((H_FULL, E_FULL), jnp.bfloat16),
                ((HP_FULL,), jnp.float32),
            ],
        )
        sh = B >= 64
        _check_mlir(
            f"b2b_{B}",
            lambda *a, _sh=sh: moe_to_attn_bridge_jax(*a, world_size=8, sharded_moe_in=_sh),
            (P(), P("model", None) if sh else P()) + (P(),) * 10,
            (P(), P(), P()),
            [
                ((B, H_FULL), jnp.float32),
                ((B, 4096), jnp.bfloat16),
                ((4096, 1024), jnp.float8_e4m3fn),
                ((1024,), jnp.float32),
                ((H_FULL,), jnp.float32),
                ((H_FULL,), jnp.float32),
                ((H_FULL,), jnp.float32),
                ((H_FULL,), jnp.float32),
                ((H_FULL, 1280), jnp.float8_e4m3fn),
                ((1280,), jnp.float32),
                ((H_FULL, 1024), jnp.float8_e4m3fn),
                ((1024,), jnp.float32),
            ],
        )

    _check_mlir(
        "l0_dqkv_80",
        lambda *a: fp8_dual_direct_qkv_epi_gate_jax(*a),
        (P(),) * 6,
        (P(), P(), P()),
        [
            ((80, H_FULL), jnp.bfloat16),
            ((H_FULL, 1280), jnp.float8_e4m3fn),
            ((1280,), jnp.float32),
            ((H_FULL, 1024), jnp.float8_e4m3fn),
            ((1024,), jnp.float32),
            ((80, 2, 128), jnp.float32),
        ],
        check_t16=False,
    )
    _check_mlir(
        "b2_dqkv_80",
        lambda *a: moe_to_attn_direct_qkv_epi_bridge_jax(*a, world_size=8),
        (P(),) * 13,
        (P(), P(None, None, "model", None, None), P(None, "model", None, None), P(None, "model")),
        [
            ((80, H_FULL), jnp.float32),
            ((80, 4096), jnp.bfloat16),
            ((4096, 1024), jnp.float8_e4m3fn),
            ((1024,), jnp.float32),
            ((H_FULL,), jnp.float32),
            ((H_FULL,), jnp.float32),
            ((H_FULL,), jnp.float32),
            ((H_FULL,), jnp.float32),
            ((H_FULL, 1280), jnp.float8_e4m3fn),
            ((1280,), jnp.float32),
            ((H_FULL, 1024), jnp.float8_e4m3fn),
            ((1024,), jnp.float32),
            ((80, 2, 128), jnp.float32),
        ],
        check_t16=False,
    )

    def _rms(x, w, eps, scale_in):
        xf = x.astype(np.float32)
        if w is not None and scale_in:
            xf = xf * w[None, :]
        inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
        y = xf * inv
        if w is not None and not scale_in:
            y = y * w[None, :]
        return y

    H_T, HP_T = 4096, 2048
    rng = np.random.default_rng(77)
    for b_t in (16, 32, 64):
        st = rng.standard_normal((b_t, H_T)).astype(np.float32)
        ga_bf = jnp.asarray(rng.standard_normal((b_t, 8 * 256)), dtype=jnp.bfloat16)
        ga_np = np.asarray(ga_bf, dtype=np.float32)
        wo = jnp.asarray(rng.standard_normal((8 * 256, H_T)) * 0.1, dtype=jnp.float8_e4m3fn)
        wos = rng.uniform(0.5, 1.5, (H_T,)).astype(np.float32)
        pa_a = rng.uniform(0.5, 1.5, (H_T,)).astype(np.float32)
        pa_b = rng.uniform(0.5, 1.5, (H_T,)).astype(np.float32)
        pff_g = rng.uniform(0.5, 1.5, (H_T,)).astype(np.float32)
        wpre = jnp.asarray(rng.standard_normal((H_T, HP_T)) * 0.1, dtype=jnp.float8_e4m3fn)
        wpre_s = rng.uniform(0.5, 1.5, (1, HP_T)).astype(np.float32)
        wrot = jnp.asarray(rng.standard_normal((H_T, 256)) * 0.1, dtype=jnp.bfloat16)
        peg = rng.uniform(0.5, 1.5, (HP_T,)).astype(np.float32)

        fn1 = jax.jit(
            jax.shard_map(
                lambda s, g, w, ws, a, b, pf, wp, wps, wr, pe: attn_to_moe_bridge_jax(
                    s, g, w, ws, a, b, pf, wp, wps, wr, pe,
                    world_size=8, pa_eps=1e-5, pre_ff_eps=1e-5, pre_ff_scale_input=False,
                    router_multiplier=0.5, pre_exp_eps=1e-5, pre_exp_scale_input=False,
                    interpret=True,
                ),
                mesh=mesh,
                in_specs=(
                    P(), P(None, "model"), P("model", None), P(), P(), P(), P(),
                    P(None, "model"), P(None, "model"), P(), P(),
                ),
                out_specs=(P(), P(), P()),
                check_vma=False,
            )
        )
        st_out, rlog_out, ein_out = fn1(
            jnp.asarray(st), ga_bf, jax.device_put(wo, NamedSharding(mesh, P("model", None))),
            jnp.asarray(wos), jnp.asarray(pa_a), jnp.asarray(pa_b), jnp.asarray(pff_g),
            jax.device_put(wpre, NamedSharding(mesh, P(None, "model"))),
            jax.device_put(jnp.asarray(wpre_s), NamedSharding(mesh, P(None, "model"))),
            wrot, jnp.asarray(peg),
        )
        o_np = (ga_np @ np.asarray(wo, dtype=np.float32)) * wos[None, :]
        st_ref = pa_a[None, :] * st + pa_b[None, :] * _rms(o_np, None, 1e-5, False)
        rin_ref = np.asarray(jnp.asarray(_rms(st_ref, pff_g, 1e-5, False), dtype=jnp.bfloat16), dtype=np.float32)
        rlog_ref = (rin_ref @ np.asarray(wrot, dtype=np.float32)) * 0.5
        pre_ref = np.asarray(jnp.asarray((rin_ref @ np.asarray(wpre, dtype=np.float32)) * wpre_s, dtype=jnp.bfloat16), dtype=np.float32)
        ein_ref = _rms(pre_ref, peg, 1e-5, False)
        assert float(np.dot(np.asarray(st_out).ravel(), st_ref.ravel()) / (np.linalg.norm(st_out) * np.linalg.norm(st_ref))) > 0.9999
        assert float(np.dot(np.asarray(rlog_out).ravel(), rlog_ref.ravel()) / (np.linalg.norm(rlog_out) * np.linalg.norm(rlog_ref))) > 0.9999
        assert float(np.dot(np.asarray(ein_out, dtype=np.float32).ravel(), ein_ref.ravel()) / (np.linalg.norm(ein_out) * np.linalg.norm(ein_ref))) > 0.9995

    for b_t, sh in ((32, False), (64, True)):
        st = rng.standard_normal((b_t, H_T)).astype(np.float32)
        moe_full = jnp.asarray(rng.standard_normal((b_t, 512)) * 0.2, dtype=jnp.bfloat16)
        moe_in = jax.device_put(moe_full, NamedSharding(mesh, P("model", None))) if sh else moe_full
        wp = jnp.asarray(rng.standard_normal((512, H_T)) * 0.1, dtype=jnp.float8_e4m3fn)
        sp = rng.uniform(0.5, 1.5, (1, H_T)).astype(np.float32)
        pfg = rng.uniform(0.5, 1.5, (H_T,)).astype(np.float32)
        pf_a = rng.uniform(0.5, 1.5, (H_T,)).astype(np.float32)
        pf_b = rng.uniform(0.5, 1.5, (H_T,)).astype(np.float32)
        nig = rng.uniform(0.5, 1.5, (H_T,)).astype(np.float32)
        wq = jnp.asarray(rng.standard_normal((H_T, 1280)) * 0.1, dtype=jnp.float8_e4m3fn)
        sq = rng.uniform(0.5, 1.5, (1280,)).astype(np.float32)
        wg = jnp.asarray(rng.standard_normal((H_T, 1024)) * 0.1, dtype=jnp.float8_e4m3fn)
        sg = rng.uniform(0.5, 1.5, (1024,)).astype(np.float32)
        in_sp = ((P(), P("model", None), P(None, "model"), P(None, "model")) if sh else (P(), P(), P(None, "model"), P(None, "model"))) + (P(),) * 8
        fn2 = jax.jit(
            jax.shard_map(
                lambda *x, _sh=sh: moe_to_attn_bridge_jax(
                    *x, world_size=8, post_ff_eps=1e-5, post_ff_scale_input=True,
                    next_in_eps=1e-5, next_in_scale_input=False, sharded_moe_in=_sh, interpret=True,
                ),
                mesh=mesh, in_specs=in_sp, out_specs=(P(), P(), P()), check_vma=False,
            )
        )
        st2_out, qkv_out, gate_out = fn2(
            jnp.asarray(st), moe_in, jax.device_put(wp, NamedSharding(mesh, P(None, "model"))),
            jax.device_put(jnp.asarray(sp), NamedSharding(mesh, P(None, "model"))),
            jnp.asarray(pfg), jnp.asarray(pf_a), jnp.asarray(pf_b), jnp.asarray(nig),
            wq, jnp.asarray(sq), wg, jnp.asarray(sg),
        )
        post_np = np.asarray(jnp.asarray((np.asarray(moe_full, dtype=np.float32) @ np.asarray(wp, dtype=np.float32)) * sp, dtype=jnp.bfloat16), dtype=np.float32)
        st2_ref = pf_a[None, :] * st + pf_b[None, :] * _rms(post_np, pfg, 1e-5, True)
        nin_ref = np.asarray(jnp.asarray(_rms(st2_ref, nig, 1e-5, False), dtype=jnp.bfloat16), dtype=np.float32)
        qkv_ref = (nin_ref @ np.asarray(wq, dtype=np.float32)) * sq[None, :]
        gate_ref = (nin_ref @ np.asarray(wg, dtype=np.float32)) * sg[None, :]
        assert float(np.dot(np.asarray(st2_out).ravel(), st2_ref.ravel()) / (np.linalg.norm(st2_out) * np.linalg.norm(st2_ref))) > 0.9999
        assert float(np.dot(np.asarray(qkv_out, dtype=np.float32).ravel(), qkv_ref.ravel()) / (np.linalg.norm(qkv_out) * np.linalg.norm(qkv_ref))) > 0.9995
        assert float(np.dot(np.asarray(gate_out, dtype=np.float32).ravel(), gate_ref.ravel()) / (np.linalg.norm(gate_out) * np.linalg.norm(gate_ref))) > 0.9995
