# SPDX-License-Identifier: Apache-2.0
"""Unit, bit-exact numerical, and Mosaic TPU7x HLO export tests for Wave-7 Prefill MoE & AG."""

from __future__ import annotations

import ast
import json
import os
import pathlib
from unittest.mock import patch

import numpy as np
import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pallas_export import tpu_info_for_cpu_export


def test_w7_prefill_flags_default_off():
    """All Wave-7 prefill flags must default to strictly off/0."""
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_AUTO_STATS.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT.get() == 0


def test_no_bare_envfield_bool_evaluation_on_w7_files():
    """Ensure all modified/added Wave-7 files call .get() on EnvField rather than bare truthiness."""
    repo = pathlib.Path(__file__).resolve().parents[5]
    rel_paths = (
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_fused_rs_nodedup.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_v2_gather_scatter.py",
        "sglang_tpu/muse_prefill_kernels/eplb.py",
        "sglang_tpu/muse_prefill_kernels/integration.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/kernel_b2.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/kernel_d.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/pipeline.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/routing_pallas.py",
    )
    checked_flags = {
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS",
        "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_EPLB_AUTO_STATS",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT",
        "SGLANG_TPU_MUSE_KERNEL_A1_AG_UNDER_D",
        "SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL",
        "SGLANG_TPU_MUSE_KERNEL_D_T64_C16",
        "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2",
    }
    for rel in rel_paths:
        tree = ast.parse((repo / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in checked_flags:
                found_get = any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "get"
                    and call.func.value is node
                    for call in ast.walk(tree)
                )
                assert found_get, f"{rel}:{node.lineno} accesses {node.attr} without .get()"


def test_pallas_routing_interpret_bit_exact_across_skew_and_pad():
    """Verify 0-ULP bit-exact equivalence of Pallas VMEM+MXU routing vs XLA reference."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _compute_local_inv_perm,
        compute_block_scatter_routing,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_global_local_pos,
        pallas_local_inv_perm_and_counts,
    )

    rng = np.random.default_rng(1119)
    ep_size = 8
    t_local = 256
    t_global = ep_size * t_local
    topk = 8

    for num_local_experts in (32, 36):
        num_experts = ep_size * num_local_experts
        # Heavy Zipf skew so one hot expert receives >15% of all rows (>256 per rank).
        probs = 1.0 / (np.arange(1, num_experts + 1, dtype=np.float64) ** 1.35)
        probs /= probs.sum()
        g_ids_np = np.empty((t_global, topk), dtype=np.int32)
        for t in range(t_global):
            g_ids_np[t] = rng.choice(num_experts, size=topk, replace=False, p=probs)

        for has_pad in (False, True):
            g_in = g_ids_np.copy()
            if has_pad:
                g_in[t_global - 300 :] = num_experts
            g_j = jnp.asarray(g_in, dtype=jnp.int32)

            for r in (0, 3, 7):
                go_val = jnp.int32(r * num_local_experts)
                my_id = jnp.int32(r)
                l_j = g_j[r * t_local : (r + 1) * t_local]
                tot_bkt = num_experts + (1 if has_pad else 0)

                with patch.dict(
                    os.environ,
                    {
                        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                        "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "0",
                    },
                ):
                    ref_inv = _compute_local_inv_perm(l_j, tot_bkt)
                    ref_lhs, ref_gs, ref_hdr, ref_trecv = compute_block_scatter_routing(
                        g_j,
                        go_val,
                        my_id,
                        ep_size=ep_size,
                        num_local_experts=num_local_experts,
                        topk=topk,
                        stable_sort=True,
                        has_pad_sentinel=has_pad,
                    )

                got_inv, got_gs, got_hdr, got_trecv = pallas_local_inv_perm_and_counts(
                    l_j,
                    g_j,
                    go_val,
                    my_id,
                    ep_size=ep_size,
                    num_local_experts=num_local_experts,
                    total_id_buckets=tot_bkt,
                    has_pad_sentinel=has_pad,
                    zero_local_start=True,
                    interpret=True,
                )
                np.testing.assert_array_equal(np.asarray(got_inv), np.asarray(ref_inv))
                np.testing.assert_array_equal(np.asarray(got_gs), np.asarray(ref_gs))
                np.testing.assert_array_equal(np.asarray(got_hdr), np.asarray(ref_hdr))
                assert int(got_trecv) == int(ref_trecv)

                pos_2d = pallas_global_local_pos(
                    g_j,
                    go_val,
                    num_local_experts=num_local_experts,
                    interpret=True,
                )
                n_flat = t_global * topk
                tok_ids_flat = jnp.arange(n_flat, dtype=jnp.int32) >> 3
                got_lhs = (
                    jnp.zeros((n_flat,), dtype=jnp.int32)
                    .at[pos_2d.reshape(-1)]
                    .set(tok_ids_flat, mode="drop", unique_indices=True)
                )
                local_total = int(
                    jnp.sum(got_gs[r * num_local_experts : (r + 1) * num_local_experts])
                )
                np.testing.assert_array_equal(
                    np.asarray(got_lhs[:local_total]),
                    np.asarray(ref_lhs[:local_total]),
                )


def test_w7_a1_v6_ag_under_d_and_pipeline_mosaic_export():
    """Verify A1_VARIANT='v6' + A1_AG_UNDER_D=1 + Wave-7 MoE pipeline export cleanly to TPU7x HLO."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh

    from sglang_tpu.muse_prefill_kernels.integration import _a1_body, _a1_first_rows
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
    ):
        for chunks_spec, expected_r0 in (("", 512), ("384,640", 384), ("256,512,256", 256)):
            r0 = _a1_first_rows(
                1024,
                "v6",
                (),
                a1_chunks=chunks_spec,
                a1_1way=True,
                rpa_layout=True,
            )
            assert r0 == expected_r0
            a1_fn = _a1_body(
                16,
                2,
                "v6",
                rpa_layout=True,
                a1_chunks=chunks_spec,
                a1_1way=True,
                ag_under_d=True,
            )
            sm_a1 = jax.jit(
                jax.shard_map(
                    a1_fn,
                    mesh=mesh,
                    in_specs=(
                        jax.sharding.PartitionSpec("model"),
                        jax.sharding.PartitionSpec("model"),
                        jax.sharding.PartitionSpec("model"),
                        jax.sharding.PartitionSpec("model"),
                        jax.sharding.PartitionSpec("model"),
                        jax.sharding.PartitionSpec("model"),
                    ),
                    out_specs=(
                        jax.sharding.PartitionSpec("model"),
                        jax.sharding.PartitionSpec("model"),
                        jax.sharding.PartitionSpec("model"),
                    ),
                    check_vma=False,
                )
            )
            hlo_a1 = (
                sm_a1.trace(
                    jax.ShapeDtypeStruct((64, r0, 8192), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8 * (1024 - r0), 8192), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8 * 8192, 2304), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8, 2304), jnp.float32),
                    jax.ShapeDtypeStruct((8 * 8192, 128), jnp.float32),
                    jax.ShapeDtypeStruct((8 * 8192, 128), jnp.float32),
                )
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )
            assert "pallas_a1" in hlo_a1
            assert "partition_id" not in hlo_a1

        try:
            for psf_mode in ("1", "2"):
                with patch.dict(
                    os.environ,
                    {
                        "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
                        "SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE": "1",
                        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                        "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": psf_mode,
                        "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "2",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                        "SGLANG_TPU_MUSE_KERNEL_D_T64_C16": "1",
                        "SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP": "1",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": (
                            "1" if psf_mode == "2" else "0"
                        ),
                    },
                ):
                    jax.clear_caches()

                    def _moe_agd_wrap(
                        a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus
                    ):
                        return jax.shard_map(
                            lambda *xs: fused_moe_full_bridge_and_ffn(
                                *xs,
                                mesh=mesh,
                                router_mode="bf16_hilo",
                                b2_layout="nstream",
                                b2_two_half=True,
                                b2_deq0_first=True,
                                hoist_routing=True,
                                d_layout="4slab",
                                d_tok_tile=64,
                                d_fp8_unpool=False,
                                in_shard_map=True,
                                b2_deq0_new_buffer=True,
                                next_in_gather_rows=512,
                                next_in_gather_static_fp8=True,
                            ),
                            mesh=mesh,
                            in_specs=(
                                jax.sharding.PartitionSpec("model", None),
                                jax.sharding.PartitionSpec("model", None),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec("model"),
                                jax.sharding.PartitionSpec("model"),
                                jax.sharding.PartitionSpec("model"),
                                jax.sharding.PartitionSpec("model"),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec("model"),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                                jax.sharding.PartitionSpec(),
                            ),
                            out_specs=(
                                jax.sharding.PartitionSpec("model", None),
                                jax.sharding.PartitionSpec("model", None),
                                jax.sharding.PartitionSpec("model", None),
                                jax.sharding.PartitionSpec("model", None),
                                jax.sharding.PartitionSpec("model", None, None),
                            ),
                            check_vma=False,
                        )(
                            a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus,
                            pfw, aa, ba, pfw,
                        )

                    lowered = jax.jit(_moe_agd_wrap).trace(
                        jax.ShapeDtypeStruct((8192, 8192), jnp.bfloat16),
                        jax.ShapeDtypeStruct((8192, 8192), jnp.float32),
                        jax.ShapeDtypeStruct((8192,), jnp.float32),
                        jax.ShapeDtypeStruct((8192,), jnp.float32),
                        jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                        jax.ShapeDtypeStruct((8192, 256), jnp.float32),
                        jax.ShapeDtypeStruct((256,), jnp.float32),
                        jax.ShapeDtypeStruct((8192, 4096), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((4096,), jnp.float32),
                        jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                        jax.ShapeDtypeStruct((256, 4096, 8192), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((256, 4096, 4096), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((256, 16, 8192), jnp.bfloat16),
                        jax.ShapeDtypeStruct((256, 16, 4096), jnp.bfloat16),
                        jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                        jax.ShapeDtypeStruct((8,), jnp.int32),
                        jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((8192,), jnp.float32),
                    ).lower(lowering_platforms=("tpu",))
                    hlo = lowered.as_text("stablehlo")
                    assert "pallas_moe_sortfree_route" in hlo
                    assert "_c16h_so" in hlo
                    assert "-rb2" in hlo
                    assert "partition_id" not in hlo
                    if psf_mode == "2":
                        assert "pallas_moe_global_pos" in hlo
                        assert "-flq1" in hlo
                        assert "stablehlo.sort" not in hlo
        finally:
            jax.clear_caches()


def test_w7_eplb_lazy_tensor_wait_for_before_free():
    """Verify _preallocate_replica_slots and _apply_layer_perm_indices flush lazy tensors via _wait_for."""
    import types
    import torch

    import sglang_tpu.expert_loading as expert_loading
    import sglang_tpu.fp8_moe as fp8_moe
    from sglang_tpu.muse_prefill_kernels.eplb import PrefillEplbManager

    events: list[str] = []
    orig_wait = expert_loading._wait_for
    orig_free = fp8_moe._free_replaced_storage

    def _spy_wait(*tensors):
        events.append(f"wait:{len(tensors)}")
        return orig_wait(*tensors)

    def _spy_free(old_t, new_t):
        events.append("free")
        return orig_free(old_t, new_t)

    def _make_layer():
        experts = torch.nn.Module()
        experts.w13_weight = torch.nn.Parameter(torch.randn(32, 8, 16), requires_grad=False)
        experts.w13_weight_scale = torch.nn.Parameter(torch.ones(32, 1, 16), requires_grad=False)
        experts.w2_weight = torch.nn.Parameter(torch.randn(32, 8, 8), requires_grad=False)
        experts.w2_weight_scale = torch.nn.Parameter(torch.ones(32, 1, 8), requires_grad=False)
        experts.num_local_experts = 32
        experts.register_buffer("_tpu_fused_rs_group_offset", torch.tensor(0, dtype=torch.int32))
        moe = torch.nn.Module()
        moe.experts = experts
        moe.gate = torch.nn.Module()
        moe.gate.weight = torch.nn.Parameter(torch.randn(64, 8), requires_grad=False)
        moe.e_score_correction_bias = torch.nn.Parameter(torch.zeros(64), requires_grad=False)
        layer = torch.nn.Module()
        layer.mlp = moe
        return layer

    decoder = torch.nn.Module()
    decoder.norm = torch.nn.Module()
    decoder.norm.weight = torch.nn.Parameter(torch.ones(8), requires_grad=False)
    decoder.layers = torch.nn.ModuleList([_make_layer(), _make_layer()])
    lw_list = [
        types.SimpleNamespace(wrouter=torch.zeros(8, 64, dtype=torch.float32))
        for _ in range(2)
    ]

    def _fake_gather(t):
        return torch.cat([t, t + 1], dim=0)

    with patch.object(expert_loading, "_wait_for", side_effect=_spy_wait), patch.object(
        fp8_moe, "_free_replaced_storage", side_effect=_spy_free
    ):
        mgr = PrefillEplbManager(
            decoder=decoder,
            weights_list=lw_list,
            gather_fn=_fake_gather,
            start=0,
            end=2,
            tp_size=2,
            tp_rank=0,
            mode="q_block",
            num_replicas=2,
        )
        # Each of 2 layers x 4 expert attrs must call _wait_for immediately before _free_replaced_storage.
        free_indices = [i for i, e in enumerate(events) if e == "free"]
        assert len(free_indices) == 8
        for idx in free_indices:
            assert idx > 0 and events[idx - 1] == "wait:1", events

        events.clear()
        mgr._apply_layer_perm_indices(
            0,
            np.arange(64, dtype=np.int64),
            np.arange(34, dtype=np.int64),
            torch.device("cpu"),
        )
        assert events == ["wait:3", "wait:1", "wait:1", "wait:1", "wait:1"]


def test_w7_qblock_closed_form_zero_while_zero_sort_and_dynamo_calib_guard():
    """Verify closed-form q_block EPLB remap emits 0 while/sort ops and Dynamo calib skips ContextVar."""
    import types
    import jax
    import jax.numpy as jnp
    import torch

    from sglang_tpu.muse_prefill_kernels.eplb import remap_zeroglue_virtual_ids
    from sglang_tpu.muse_prefill_kernels.integration import MusePrefillKernels

    lowered = (
        jax.jit(
            lambda ids: remap_zeroglue_virtual_ids(
                ids,
                jnp.int32(0),
                ep_size=8,
                num_local_experts=36,
                mode="q_block",
                quantum=128,
            )
        )
        .trace(jax.ShapeDtypeStruct((8192, 8), jnp.int32))
        .lower()
    )
    hlo = lowered.as_text("stablehlo")
    assert "stablehlo.while" not in hlo
    assert "stablehlo.sort" not in hlo

    dummy_self = types.SimpleNamespace(
        eplb_counts=torch.zeros(2, 256, dtype=torch.int64),
        eplb_calib_step=torch.zeros(1, dtype=torch.int32),
        _consume_eplb_calib_hists=lambda _: (_ for _ in ()).throw(
            AssertionError("should not run under is_compiling")
        ),
    )
    with patch("torch.compiler.is_compiling", return_value=True), patch(
        "sglang_tpu.compilation.direct_compile.insert_callback",
        side_effect=AssertionError("insert_callback must not run under TorchDynamo"),
    ):
        MusePrefillKernels._finish_eplb_calib_step(
            dummy_self,
            [torch.ones(256, dtype=torch.int32), torch.full((256,), 2, dtype=torch.int32)],
        )
    assert int(dummy_self.eplb_calib_step.item()) == 1
    assert int(dummy_self.eplb_counts[0, 0].item()) == 1
    assert int(dummy_self.eplb_counts[1, 0].item()) == 2


def test_w7_finer_bucket_7168_and_packed_b2_scale_unpack():
    """Verify #1102 & #1133 fixes: lossless u16 BF16 byte-split round-trip across sc >= 1.0 (incl. 1.5, 100.0)."""
    import jax
    import jax.numpy as jnp

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _unpack_ag_i8_and_scale,
    )

    sc_ref = jnp.array([[0.03125], [0.25], [1.5], [100.0]], dtype=jnp.bfloat16)
    u16 = jax.lax.bitcast_convert_type(sc_ref, jnp.uint16).astype(jnp.int32)
    lo_b = (u16 & jnp.int32(0xFF)).astype(jnp.int8)
    hi_b = jax.lax.shift_right_logical(u16, 8).astype(jnp.int8)
    lane = jnp.arange(128, dtype=jnp.int32)[None, :]
    ps_bf16 = jnp.where(lane < 64, lo_b, hi_b)
    fake_ag = jnp.concatenate(
        [jnp.ones((4, 4096), dtype=jnp.int8) * 4, ps_bf16], axis=1
    )
    for b2_i8pack_flag in (False, True):
        data_o, sc_o = _unpack_ag_i8_and_scale(
            fake_ag, None, expert_axis="model", b2_i8pack=b2_i8pack_flag, he_dim=4096
        )
        assert data_o.shape == (4, 4096)
        np.testing.assert_array_equal(np.asarray(sc_o), np.asarray(sc_ref))


def test_w7_pallas_qblock_eplb_remap_bit_exact_and_tpu7x_export():
    """Verify pallas_qblock_remap_virtual_ids is 0-ULP bit-exact vs XLA across T=4096, 8192, 16384 and exports on TPU7x."""
    import jax
    import jax.numpy as jnp

    from sglang_tpu.muse_prefill_kernels.eplb import remap_zeroglue_virtual_ids
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_qblock_remap_virtual_ids,
    )

    rng = np.random.default_rng(20261125)
    for t_tot, q_m in ((4096, 128), (8192, 128), (16384, 256)):
        probs = 1.0 / (np.arange(1, 257, dtype=np.float64) ** 1.35)
        probs /= probs.sum()
        g_np = rng.choice(256, size=(t_tot, 8), p=probs).astype(np.int32)
        g_np[-128:, :] = 256  # pad sentinel rows
        g_jnp = jnp.asarray(g_np)
        for n_rep in (2, 4):
            e_loc = 32 + n_rep
            for rk in (0, 5):
                v_ref_g, v_ref_l = remap_zeroglue_virtual_ids(
                    g_jnp,
                    jnp.int32(rk),
                    ep_size=8,
                    num_local_experts=e_loc,
                    mode="q_block",
                    quantum=q_m,
                )
                v_pal_g, v_pal_l = pallas_qblock_remap_virtual_ids(
                    g_jnp,
                    jnp.int32(rk),
                    ep_size=8,
                    num_local_experts=e_loc,
                    quantum=q_m,
                    interpret=True,
                )
                np.testing.assert_array_equal(np.asarray(v_pal_g), np.asarray(v_ref_g))
                np.testing.assert_array_equal(np.asarray(v_pal_l), np.asarray(v_ref_l))

    if len(jax.devices("cpu")) >= 8:
        with tpu_info_for_cpu_export(), patch.dict(
            os.environ, {"SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "1"}
        ):
            lowered = (
                jax.jit(
                    lambda ids: remap_zeroglue_virtual_ids(
                        ids,
                        jnp.int32(3),
                        ep_size=8,
                        num_local_experts=36,
                        mode="q_block",
                        quantum=128,
                    )
                )
                .trace(jax.ShapeDtypeStruct((8192, 8), jnp.int32))
                .lower(lowering_platforms=("tpu",))
            )
            hlo = lowered.as_text("stablehlo")
            assert "pallas_qblock_eplb_remap" in hlo


def test_w7_eplb_fp4_direct_copy_avoids_view_uint8_write():
    """Verify eplb.py avoids .view(uint8).copy_() (2x EXTEND HBM fix) across prealloc & rebalance."""
    import types
    import torch

    from sglang_tpu.muse_prefill_kernels.eplb import PrefillEplbManager

    repo = pathlib.Path(__file__).resolve().parents[5]
    eplb_code_lines = [
        line
        for line in (repo / "sglang_tpu/muse_prefill_kernels/eplb.py")
        .read_text()
        .splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert all(".view(torch.uint8).copy_(" not in line for line in eplb_code_lines)

    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is None:
        return
    torch.manual_seed(7)
    u13 = torch.randint(0, 255, (32, 8, 16), dtype=torch.uint8)
    u2 = torch.randint(0, 255, (32, 8, 8), dtype=torch.uint8)
    experts = torch.nn.Module()
    experts.w13_weight = torch.nn.Parameter(u13.view(fp4).clone(), requires_grad=False)
    experts.w13_weight_scale = torch.nn.Parameter(
        torch.ones(32, 1, 16, dtype=torch.bfloat16), requires_grad=False
    )
    experts.w2_weight = torch.nn.Parameter(u2.view(fp4).clone(), requires_grad=False)
    experts.w2_weight_scale = torch.nn.Parameter(
        torch.ones(32, 1, 8, dtype=torch.bfloat16), requires_grad=False
    )
    experts.num_local_experts = 32
    experts.register_buffer("_tpu_fused_rs_group_offset", torch.tensor(0, dtype=torch.int32))
    moe = torch.nn.Module()
    moe.experts = experts
    moe.gate = torch.nn.Module()
    moe.gate.weight = torch.nn.Parameter(torch.randn(64, 8), requires_grad=False)
    moe.e_score_correction_bias = torch.nn.Parameter(torch.zeros(64), requires_grad=False)
    layer = torch.nn.Module()
    layer.mlp = moe
    decoder = torch.nn.Module()
    decoder.norm = torch.nn.Module()
    decoder.norm.weight = torch.nn.Parameter(torch.ones(8), requires_grad=False)
    decoder.layers = torch.nn.ModuleList([layer])
    lw_list = [types.SimpleNamespace(wrouter=torch.zeros(8, 64, dtype=torch.float32))]

    def _gather_u8(t):
        return torch.cat([t, t], dim=0)

    mgr = PrefillEplbManager(
        decoder=decoder,
        weights_list=lw_list,
        gather_fn=_gather_u8,
        start=0,
        end=1,
        tp_size=2,
        tp_rank=0,
        mode="q_block",
        num_replicas=2,
    )
    assert tuple(experts.w13_weight.shape) == (34, 8, 16)
    assert experts.w13_weight.dtype == fp4
    assert experts.w13_weight._base is None
    torch.testing.assert_close(experts.w13_weight.data.view(torch.uint8)[:32], u13)
    torch.testing.assert_close(experts.w13_weight.data.view(torch.uint8)[32:34], u13[:2])
    mgr._apply_layer_perm_indices(
        0,
        np.arange(64, dtype=np.int64),
        np.arange(34, dtype=np.int64),
        torch.device("cpu"),
    )
    assert experts.w13_weight._base is None





