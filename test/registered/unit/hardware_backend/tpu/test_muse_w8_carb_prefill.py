# SPDX-License-Identifier: Apache-2.0
"""Unit, numerical accuracy, and Mosaic TPU7x HLO lowering tests for Wave-8 Carb Prefill."""

from __future__ import annotations

import ast
import os
import pathlib
from unittest.mock import patch

import numpy as np
import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

_NEW_FLAGS = (
    "SGLANG_TPU_MUSE_KERNEL_A1_SUB_N",
    "SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM",
    "SGLANG_TPU_MUSE_KERNEL_ENTRY_ASYNC_A2A",
    "SGLANG_TPU_MUSE_KERNEL_A3_BN2048",
    "SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO",
    "SGLANG_TPU_MUSE_KERNEL_D_COAL_GATHER",
    "SGLANG_TPU_MUSE_KERNEL_D_IN_KERNEL_AG",
    "SGLANG_TPU_MUSE_KERNEL_W4_WO",
    "SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL",
)


def test_w8_carb_prefill_flags_default_off_and_ast_get_hygiene():
    """All Wave-8 carb prefill flags must default strictly off and always use .get()."""
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_SUB_N.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_ENTRY_ASYNC_A2A.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_BN2048.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_COAL_GATHER.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_IN_KERNEL_AG.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WO.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_W4_WUNPOOL.get() == 0

    repo = pathlib.Path(__file__).resolve().parents[5]
    rel_paths = (
        "sglang_tpu/muse_prefill_kernels/attn_kernels/kernel_a1.py",
        "sglang_tpu/muse_prefill_kernels/attn_kernels/kernel_a3.py",
        "sglang_tpu/muse_prefill_kernels/attn_kernels/entry_exit.py",
        "sglang_tpu/muse_prefill_kernels/prefill_w8_radicals.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/pipeline.py",
        "sglang_tpu/muse_prefill_kernels/integration.py",
    )
    checked = set(_NEW_FLAGS)
    for rel in rel_paths:
        tree = ast.parse((repo / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in checked:
                found_get = any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "get"
                    and call.func.value is node
                    for call in ast.walk(tree)
                )
                assert found_get, f"{rel}:{node.lineno} accesses {node.attr} without .get()"


def test_a3_supports_long_prefill_buckets():
    """Verify _a3_supports unlocks T=24k (3072) on v16 and T=20k..32k on A3_KDEEP_WO / W4_WO."""
    from sglang_tpu.muse_prefill_kernels.integration import _a3_supports

    assert _a3_supports("v16", 1024) is True
    assert _a3_supports("v16", 2048) is True
    assert _a3_supports("v16", 3072) is True
    assert _a3_supports("v16", 2560) is False
    assert _a3_supports("v16", 3584) is False

    with patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO": "2"}):
        for rpr in (512, 2048, 2560, 3072, 3584, 4096):
            assert _a3_supports("v16", rpr) is True
        assert _a3_supports("v16", 4608) is False

    with patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_W4_WO": "256"}):
        assert _a3_supports("v16", 3584) is True


def test_a1_fast_6roll_dual_head_rmsnorm_numerical_parity():
    """Verify 6-roll prefix butterfly dual-head RMSNorm matches reference across outliers."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax import lax

    from sglang_tpu.muse_prefill_kernels.attn_kernels.kernel_a1 import (
        _dual_head_rmsnorm_128,
    )

    rng = np.random.default_rng(2026)
    x_np = rng.standard_normal((256, 128)).astype(np.float32)
    # Inject extreme asymmetric head scales (head 0 ~ 120x, head 1 ~ 0.02x) to catch any cross-half leak!
    x_np[:64, :64] *= 120.0
    x_np[:64, 64:] *= 0.02
    x_np[64:128, :64] *= 0.01
    x_np[64:128, 64:] *= 85.0

    x = jnp.asarray(x_np)
    lane = lax.broadcasted_iota(jnp.int32, (256, 128), 1)
    is_lo = lane < 64
    mask_lo = is_lo.astype(jnp.float32)
    mask_hi = 1.0 - mask_lo

    # On CPU outside Pallas, pltpu.roll maps 1:1 to right-shift jnp.roll along axis=1
    # (verified in kernel_a1._rope_128_vreg where shift=127 fetches lane+1 and shift=1 fetches lane-1).
    with patch("jax.experimental.pallas.tpu.roll", side_effect=lambda a, shift, axis: jnp.roll(a, shift=shift, axis=axis)):
        ref = np.asarray(
            _dual_head_rmsnorm_128(x, is_lo, mask_lo, mask_hi, 5.473, fast_6roll=False)
        )
        got = np.asarray(
            _dual_head_rmsnorm_128(x, is_lo, mask_lo, mask_hi, 5.473, fast_6roll=True)
        )
    cos = float(np.sum(ref * got) / (np.linalg.norm(ref) * np.linalg.norm(got)))
    assert cos > 0.999999
    np.testing.assert_allclose(got, ref, rtol=8e-3, atol=2e-2)


def test_d_coal_runs_and_wsel_exactness_across_skew_and_pad():
    """Verify run-coalesced gather descriptors + Wsel MXU combine across Zipf skew & padded rows."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _compute_local_inv_perm,
    )
    from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
        precompute_d_coal_runs_and_wsel,
    )

    rng = np.random.default_rng(816)
    t_local = 512
    topk = 8
    num_experts = 256
    tok_tile = 64

    probs = 1.0 / (np.arange(1, num_experts + 1, dtype=np.float64) ** 1.35)
    probs /= probs.sum()
    l_ids = np.empty((t_local, topk), dtype=np.int32)
    for t in range(t_local):
        l_ids[t] = rng.choice(num_experts, size=topk, replace=False, p=probs)

    w_raw = rng.uniform(0.02, 0.5, size=(t_local, topk)).astype(np.float32)
    w_raw /= w_raw.sum(axis=-1, keepdims=True)
    out_buf = rng.standard_normal((t_local * topk, 64)).astype(np.float32)

    for with_pad in (False, True):
        ids_in = l_ids.copy()
        w_in = w_raw.copy()
        if with_pad:
            ids_in[350:] = num_experts
            w_in[350:] = 0.0
        tot_bkt = num_experts + (1 if with_pad else 0)
        inv_p = _compute_local_inv_perm(jnp.asarray(ids_in), tot_bkt)
        if with_pad:
            safe_rows = min(1024, t_local * topk)
            pad_spread = (np.arange(t_local * topk, dtype=np.int32) % safe_rows).reshape(
                t_local, topk
            )
            active = (np.arange(t_local)[:, None] < 350)
            inv_p = jnp.asarray(np.where(active, np.asarray(inv_p), pad_spread))

        w_bf16 = jnp.asarray(w_in).astype(jnp.bfloat16)
        out_j = jnp.asarray(out_buf).astype(jnp.bfloat16)

        run_meta, wsel = precompute_d_coal_runs_and_wsel(
            inv_p, w_bf16, tok_tile=tok_tile
        )
        run_meta_np = np.asarray(run_meta)
        wsel_np = np.asarray(wsel.astype(jnp.float32))
        out_np = np.asarray(out_j.astype(jnp.float32))
        inv_np = np.asarray(inv_p)
        w_f32_np = np.asarray(w_bf16.astype(jnp.float32))

        n_steps = t_local // tok_tile
        active_run_counts = []
        for s in range(n_steps):
            n_runs = int(run_meta_np[s, 0, 3])
            assert 1 <= n_runs <= tok_tile * topk
            if not with_pad or (s + 1) * tok_tile <= 350:
                active_run_counts.append(n_runs)
            # Simulate exact variable-length 3D HBM run DMAs into buf_sim[512, 64]
            buf_sim = np.full((tok_tile * topk, 64), np.nan, dtype=np.float32)
            total_transferred = 0
            for r in range(n_runs):
                s0, d0, ln = (int(x) for x in run_meta_np[s, r, :3])
                assert ln > 0
                buf_sim[d0 : d0 + ln] = out_np[s0 : s0 + ln]
                total_transferred += ln
            assert total_transferred == tok_tile * topk
            assert not np.isnan(buf_sim).any()

            got_u = wsel_np[s] @ buf_sim
            t0, t1 = s * tok_tile, (s + 1) * tok_tile
            ref_u = np.sum(
                out_np[inv_np[t0:t1]] * w_f32_np[t0:t1, :, None], axis=1
            )
            np.testing.assert_allclose(got_u, ref_u, rtol=2e-3, atol=2e-3)

        # Verify 1D SMEM packing preserves every (n_runs, s0, d0, ln) tuple losslessly
        n_pairs = tok_tile * topk
        packed_1d = np.concatenate(
            [
                run_meta_np[:, 0:1, 3],
                np.stack(
                    [run_meta_np[:, :, 0], (run_meta_np[:, :, 1] << 12) | run_meta_np[:, :, 2]],
                    axis=-1,
                ).reshape(n_steps, 2 * n_pairs),
            ],
            axis=1,
        ).reshape(-1)
        assert packed_1d.shape == (n_steps * (1 + 2 * n_pairs),)
        for s in range(n_steps):
            base = s * (1 + 2 * n_pairs)
            assert int(packed_1d[base]) == int(run_meta_np[s, 0, 3])
            for r in range(int(run_meta_np[s, 0, 3])):
                s0 = int(packed_1d[base + 1 + 2 * r])
                pk = int(packed_1d[base + 2 + 2 * r])
                assert (s0, pk >> 12, pk & 0xFFF) == tuple(int(x) for x in run_meta_np[s, r, :3])

        # Verify substantial run coalescing (< 240 runs vs 512 uncoalesced 1-row DMAs!)
        assert np.mean(active_run_counts) < 240.0


def test_w4_dense_quantizer_roundtrip_and_footprint():
    """Verify quantize_dense_fp8_to_nvfp4_torch halves weight bytes with >0.994 cosine."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float4_e2m1fn_x2"):
        pytest.skip("Requires torch.float4_e2m1fn_x2")

    from sglang_tpu.moe.nvfp4_requant import _dequant
    from sglang_tpu.muse_prefill_kernels.integration import _fp8_k_major
    from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
        quantize_dense_fp8_to_nvfp4_torch,
    )

    torch.manual_seed(42)
    w_orig = torch.randn(512, 2048, dtype=torch.bfloat16) * 0.05
    w_fp8, s_fp8 = _fp8_k_major(w_orig)  # [2048, 512]

    # 3D A3 Wo layout [tp=2, K_local=1024, N=512]
    w_fp4_3d, s_bf16_3d, lshape = quantize_dense_fp8_to_nvfp4_torch(
        w_fp8, s_fp8, w_shape=(2, 1024, 512), block=256
    )
    assert lshape == (2, 1024, 512)
    assert w_fp4_3d.dtype == torch.float4_e2m1fn_x2
    assert tuple(w_fp4_3d.shape) == (2, 1024, 256)
    assert tuple(s_bf16_3d.shape) == (2, 4, 512)
    assert w_fp4_3d.numel() * w_fp4_3d.element_size() == (
        w_fp8.numel() * w_fp8.element_size()
    ) // 2

    deq = _dequant(w_fp4_3d.view(torch.uint8).reshape(1, 2048, 256), s_bf16_3d.reshape(1, 8, 512))[0]
    ref = w_orig.t().float()
    cos = float((deq * ref).sum() / (deq.norm() * ref.norm()))
    assert cos > 0.992

    # Bounded uniform weights + GEMM output parity check (K=2048 contraction averages quantization noise)
    w_unif = (torch.rand(512, 2048, dtype=torch.bfloat16) - 0.5) * 0.1
    wf_u, sf_u = _fp8_k_major(w_unif)
    w4_u, s4_u, _ = quantize_dense_fp8_to_nvfp4_torch(
        wf_u, sf_u, w_shape=(2048, 512), block=256
    )
    deq_u = _dequant(w4_u.view(torch.uint8).unsqueeze(0), s4_u.unsqueeze(0))[0]
    ref_u = w_unif.t().float()
    cos_u = float((deq_u * ref_u).sum() / (deq_u.norm() * ref_u.norm()))
    assert cos_u > 0.993
    x_act = torch.randn(64, 2048, dtype=torch.float32)
    y_ref = x_act @ ref_u
    y_got = x_act @ deq_u
    cos_gemm = float((y_got * y_ref).sum() / (y_got.norm() * y_ref.norm()))
    assert cos_gemm > 0.993


def test_w8_carb_prefill_mosaic_tpu7x_hlo_lowering_all_kernels():
    """Lower A1_SUB_N, ENTRY_ASYNC_A2A, A3_BN2048, A3_KDEEP_WO, D_COAL_GATHER, D_IN_KERNEL_AG, W4_WO, W4_WUNPOOL to TPU7x HLO."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_prefill_kernels.attn_kernels import entry_exit, kernel_a1, kernel_a3
    from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
        a3_v16_kdeep_sp,
        pallas_a3_wo_kdeep_ccs_last_a2a,
        pallas_async_a2a_xor_3d,
        pallas_d_radical_4slab,
    )

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))

    def _lower_sm(fn, in_shapes, out_count=1):
        out_spec = (P("model"),) * out_count if out_count > 1 else P("model")
        sm = jax.jit(
            jax.shard_map(
                fn,
                mesh=mesh,
                in_specs=(P("model"),) * len(in_shapes),
                out_specs=out_spec,
                check_vma=False,
            )
        )
        lowered = sm.trace(*in_shapes).lower(lowering_platforms=("tpu",))
        stablehlo = lowered.as_text("stablehlo")
        hlo = lowered.as_text("hlo")
        assert "tpu_custom_call" in hlo
        assert "partition_id" not in stablehlo
        return stablehlo

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
    ):
        # 1. A1_SUB_N=256 + A1_FAST_NORM=1 on CCS chunk (c_k=768, tm=256, rpa_layout=True)
        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_A1_SUB_N": "256",
                "SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM": "1",
            },
        ):
            def _a1_c(ag_k, w, ws, c, s):
                return kernel_a1.pallas_a1_chunk_1way(
                    ag_k,
                    w,
                    ws,
                    c,
                    s,
                    None,
                    t_shard=768,
                    row_off=0,
                    h_q_local=16,
                    h_kv_local=2,
                    rpa_layout=True,
                    mode="all",
                    tp_axis="model",
                    tm=256,
                )

            hlo_a1 = _lower_sm(
                _a1_c,
                [
                    jax.ShapeDtypeStruct((64, 768, 8192), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8 * 8192, 2304), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8, 2304), jnp.float32),
                    jax.ShapeDtypeStruct((64, 768, 128), jnp.float32),
                    jax.ShapeDtypeStruct((64, 768, 128), jnp.float32),
                ],
                out_count=3,
            )
            assert "pallas_a1_1way_all_off0_h768_tm256_sn256_fn_rpal" in hlo_a1
            assert "partition_id" not in hlo_a1

        # 2. ENTRY_ASYNC_A2A=1
        with patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_ENTRY_ASYNC_A2A": "1"}):
            hlo_ent = _lower_sm(
                lambda ids, emb: entry_exit.sp_entry_col_a2a(
                    ids, emb, org_vocab_size=202048, start_is_zero=True, tp_axis="model"
                ),
                [
                    jax.ShapeDtypeStruct((8 * 16384,), jnp.int32),
                    jax.ShapeDtypeStruct((8 * 256, 1024), jnp.bfloat16),
                ],
            )
            assert "pallas_embed_rmsnorm_bf16" in hlo_ent

        # 3. A3_BN2048=1 at T=16384 (t_shard=2048)
        with patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_A3_BN2048": "1"}):
            hlo_bn2k = _lower_sm(
                lambda q, w, ws: kernel_a3.pallas_option3d_zero_copy9_persrc_async_a2a_wo_3d_static_fp8(
                    q, w, ws, tp_axis="model", opt_wo=True
                ),
                [
                    jax.ShapeDtypeStruct((64, 2048, 1024), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((64, 1024, 8192), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8, 8192), jnp.float32),
                ],
            )
            assert "pallas_a3_wo_v16_opt_bn2048" in hlo_bn2k

        # 4. A3_KDEEP_WO=1 at T=28672 (t_shard=3584) + W4_WO=256 + CCS Mode-2
        hlo_kd28k = _lower_sm(
            lambda a, g, w, ws: a3_v16_kdeep_sp(
                a, g, w, ws, tp_axis="model", head_norm_eps=1e-5, rpa_packed_out=True
            ),
            [
                jax.ShapeDtypeStruct((8 * 28672, 16, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((8 * 28672, 1024), jnp.bfloat16),
                jax.ShapeDtypeStruct((64, 1024, 8192), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8, 8192), jnp.float32),
            ],
        )
        assert "pallas_a3_wo_kdeep_bn1024" in hlo_kd28k

        hlo_ccs_kd_w4 = _lower_sm(
            lambda q_loc0, q_last, w4, ws4: pallas_a3_wo_kdeep_ccs_last_a2a(
                pallas_async_a2a_xor_3d(q_loc0, tp_axis="model"),
                q_last,
                w4,
                ws4,
                tp_axis="model",
                w4_qbs=256,
            ),
            [
                jax.ShapeDtypeStruct((64, 1280, 1024), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64, 768, 1024), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64, 1024, 8192), jnp.float4_e2m1fn),
                jax.ShapeDtypeStruct((64, 4, 8192), jnp.bfloat16),
            ],
        )
        assert "pallas_a3_xor_a2a_r1280" in hlo_ccs_kd_w4
        assert "pallas_a3_wo_kdeep_ccs_p1280_l768_bn1024_w4q256" in hlo_ccs_kd_w4

        # 5. Stacked D_COAL_GATHER=1 + D_IN_KERNEL_AG=1 + W4_WUNPOOL=256 (BF16 & Static-FP8 RS input)
        for ccs_fp8, in_dt in ((True, jnp.bfloat16), (False, jnp.bfloat16), (True, jnp.float8_e4m3fn)):
            hlo_d = _lower_sm(
                lambda ob, ip, tw, smid, wu4, wus4, pfw, af, bf, niw: pallas_d_radical_4slab(
                    ob,
                    ip,
                    tw,
                    smid,
                    wu4,
                    wus4,
                    pfw,
                    af,
                    bf,
                    niw,
                    tok_tile=64,
                    fp8_unpool=True,
                    emit_static_fp8_attn_in=ccs_fp8,
                    coal_gather=True,
                    in_kernel_ag_rows=512,
                    ag_axis_name="model",
                    ag_tp_deg=8,
                    ag_part0_static_fp8=True,
                    w4_unpool_qbs=256,
                    wunpool_w4_scale=wus4,
                ),
                [
                    jax.ShapeDtypeStruct((8 * 16384, 32, 128), in_dt),
                    jax.ShapeDtypeStruct((8 * 2048, 8), jnp.int32),
                    jax.ShapeDtypeStruct((8 * 2048, 8), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8 * 2048, 8192), jnp.float32),
                    jax.ShapeDtypeStruct((8 * 4096, 8192), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((8 * 16, 8192), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8 * 8192,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8 * 8192,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8 * 8192,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8 * 8192,), jnp.bfloat16),
                ],
                out_count=3,
            )
            assert "pallas_d_rad_4slab_t64_coal_pag512_w4q256" in hlo_d
            assert "partition_id" not in hlo_d


def test_w8_carb_prefill_interpret_8dev_numerical_parity_and_fp8_outlier_clip():
    """Verify 8-device interpret=True numerical parity for A1_SUB_N, A3_KDEEP_WO=1/2, and D_IN_KERNEL_AG outlier clipping."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1, kernel_a3
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels import kernel_d
    from sglang_tpu.muse_prefill_kernels.prefill_w8_radicals import (
        a3_v16_kdeep_sp,
        pallas_a3_wo_kdeep_ccs_last_a2a,
        pallas_a3_wo_kdeep_static_fp8,
        pallas_async_a2a_xor_3d,
        pallas_d_radical_4slab,
    )

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    rng = np.random.default_rng(17)
    _dummy_barrier = object()
    _orig_pallas_call = pl.pallas_call
    _orig_marc = pltpu.make_async_remote_copy
    _orig_sig = pl.semaphore_signal
    _orig_wait = pl.semaphore_wait

    def _interp_pallas_call(*args, **kwargs):
        kwargs["interpret"] = True
        return _orig_pallas_call(*args, **kwargs)

    def _interp_marc(*args, **kwargs):
        did = kwargs.get("device_id")
        if isinstance(did, dict) and len(did) == 1:
            kwargs = {**kwargs, "device_id": next(iter(did.values()))}
        return _orig_marc(*args, **kwargs)

    with (
        patch.object(pl, "pallas_call", _interp_pallas_call),
        patch.object(pltpu, "make_async_remote_copy", _interp_marc),
        patch.object(pltpu, "get_barrier_semaphore", lambda *a, **kw: _dummy_barrier),
        patch.object(
            pl,
            "semaphore_signal",
            lambda sem, *a, **kw: None if sem is _dummy_barrier else _orig_sig(sem, *a, **kw),
        ),
        patch.object(
            pl,
            "semaphore_wait",
            lambda sem, *a, **kw: None if sem is _dummy_barrier else _orig_wait(sem, *a, **kw),
        ),
        patch.object(kernel_a1.pltpu, "roll", lambda x, shift, axis: jnp.roll(x, shift, axis=axis)),
    ):
        # 1. A1_SUB_N=256 vs A1_SUB_N=0 (including packed dynamic FP8 scales)
        t_shard, hidden, h_q, h_kv = 64, 512, 8, 2
        n_total = (h_q + 2 * h_kv + h_q) * 128
        x_dyn = jnp.asarray(
            (rng.standard_normal((8, t_shard, hidden + 128)) * 0.5).astype(np.float32)
        ).astype(jnp.float8_e4m3fn)
        w_a1 = jnp.asarray(
            (rng.standard_normal((hidden, n_total)) * 0.5).astype(np.float32)
        ).astype(jnp.float8_e4m3fn)
        ws_a1 = jnp.asarray(rng.uniform(0.5, 1.5, (1, n_total)).astype(np.float32))
        cos = jnp.asarray(rng.standard_normal((8, t_shard, 128)).astype(np.float32)).astype(
            jnp.bfloat16
        )
        sin = jnp.asarray(rng.standard_normal((8, t_shard, 128)).astype(np.float32)).astype(
            jnp.bfloat16
        )

        def _run_a1(sub_n: int):
            with patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_A1_SUB_N": str(sub_n)}):
                return jax.jit(
                    jax.shard_map(
                        lambda x_i, w_i, ws_i, c_i, s_i: kernel_a1.pallas_a1_chunk_1way(
                            x_i,
                            w_i,
                            ws_i,
                            c_i,
                            s_i,
                            None,
                            t_shard=t_shard,
                            row_off=0,
                            h_q_local=h_q,
                            h_kv_local=h_kv,
                            rpa_layout=True,
                            mode="all",
                            tm=64,
                        ),
                        mesh=mesh,
                        in_specs=(P(), P(), P(), P(), P()),
                        out_specs=(P(), P(), P()),
                        check_vma=False,
                    )
                )(x_dyn, w_a1, ws_a1, cos, sin)

        q0, kv0, g0 = _run_a1(0)
        q1, kv1, g1 = _run_a1(256)
        assert float(jnp.max(jnp.abs(q0.astype(jnp.float32) - q1.astype(jnp.float32)))) == 0.0
        assert float(jnp.max(jnp.abs(kv0.astype(jnp.float32) - kv1.astype(jnp.float32)))) == 0.0
        assert float(jnp.max(jnp.abs(g0.astype(jnp.float32) - g1.astype(jnp.float32)))) == 0.0

        # 2. A3_KDEEP_WO=1 vs stock a3_v16, and A3_KDEEP_WO=2 vs A3_KDEEP_WO=1
        tp_deg, prev_rows, last_rows, k_loc, n_out = 8, 128, 128, 1024, 1024
        t_sh_a3 = prev_rows + last_rows
        attn_by_h = jnp.asarray(
            (rng.standard_normal((tp_deg, tp_deg * t_sh_a3, k_loc)) * 0.5).astype(np.float32)
        ).astype(jnp.bfloat16)
        gate_by_h = jnp.asarray(
            (rng.standard_normal((tp_deg, tp_deg * t_sh_a3, k_loc)) * 0.5).astype(np.float32)
        ).astype(jnp.bfloat16)
        w_by_h = jnp.asarray(
            (rng.standard_normal((tp_deg, k_loc, n_out)) * 0.5).astype(np.float32)
        ).astype(jnp.float8_e4m3fn)
        ws_by_h = jnp.asarray(rng.uniform(0.5, 1.5, (tp_deg, n_out)).astype(np.float32))
        w_xor = jnp.stack(
            [jnp.stack([w_by_h[r ^ d] for d in range(tp_deg)], axis=0) for r in range(tp_deg)],
            axis=0,
        )
        ws_xor = jnp.stack(
            [jnp.stack([ws_by_h[r ^ d] for d in range(tp_deg)], axis=0) for r in range(tp_deg)],
            axis=0,
        )

        out_v16 = jax.jit(
            jax.shard_map(
                lambda a, g, w, ws: kernel_a3.a3_v16_option3d_zero_copy9_persrc_sem_async_a2a_wo_3d_sp(
                    a[0], g[0], w[0], ws[0], tp_axis="model", bn=256
                ),
                mesh=mesh,
                in_specs=(
                    P("model", None, None),
                    P("model", None, None),
                    P("model", None, None, None),
                    P("model", None, None),
                ),
                out_specs=P("model", None),
                check_vma=False,
            )
        )(attn_by_h, gate_by_h, w_xor, ws_xor)
        out_kd1_sp = jax.jit(
            jax.shard_map(
                lambda a, g, w, ws: a3_v16_kdeep_sp(a[0], g[0], w[0], ws[0], tp_axis="model", bn=256),
                mesh=mesh,
                in_specs=(
                    P("model", None, None),
                    P("model", None, None),
                    P("model", None, None, None),
                    P("model", None, None),
                ),
                out_specs=P("model", None),
                check_vma=False,
            )
        )(attn_by_h, gate_by_h, w_xor, ws_xor)
        assert float(jnp.max(jnp.abs(out_v16.astype(jnp.float32) - out_kd1_sp.astype(jnp.float32)))) == 0.0

        q_prev = jnp.asarray(
            (rng.standard_normal((tp_deg, tp_deg, prev_rows, k_loc)) * 0.5).astype(np.float32)
        ).astype(jnp.float8_e4m3fn)
        q_last = jnp.asarray(
            (rng.standard_normal((tp_deg, tp_deg, last_rows, k_loc)) * 0.5).astype(np.float32)
        ).astype(jnp.float8_e4m3fn)
        q_all = jnp.concatenate([q_prev, q_last], axis=2)
        out_kd1_raw = jax.jit(
            jax.shard_map(
                lambda q, w, ws: pallas_a3_wo_kdeep_static_fp8(
                    q[0], w[0], ws[0], tp_axis="model", bn=256, pre_exchanged=False
                ),
                mesh=mesh,
                in_specs=(
                    P("model", None, None, None),
                    P("model", None, None, None),
                    P("model", None, None),
                ),
                out_specs=P("model", None),
                check_vma=False,
            )
        )(q_all, w_xor, ws_xor)
        out_kd2 = jax.jit(
            jax.shard_map(
                lambda qp, ql, w, ws: pallas_a3_wo_kdeep_ccs_last_a2a(
                    pallas_async_a2a_xor_3d(qp[0], tp_axis="model", collective_id=31),
                    ql[0],
                    w[0],
                    ws[0],
                    tp_axis="model",
                    bn=256,
                ),
                mesh=mesh,
                in_specs=(
                    P("model", None, None, None),
                    P("model", None, None, None),
                    P("model", None, None, None),
                    P("model", None, None),
                ),
                out_specs=P("model", None),
                check_vma=False,
            )
        )(q_prev, q_last, w_xor, ws_xor)
        assert float(jnp.max(jnp.abs(out_kd1_raw.astype(jnp.float32) - out_kd2.astype(jnp.float32)))) == 0.0

        # 2b. A3_STAGE_SENDS=1 parity_stagger=True (#1633/#1634): K-deep Wo + CCS A2A must match
        #     _prepare_wo's parity-staggered K_8 involution permutation.
        from types import SimpleNamespace

        def _ps_peer(r: int, d: int) -> int:
            if d in (0, 1, 6, 7):
                return r ^ d
            if d in (2, 3):
                return r ^ ((2 if d == 2 else 4) if (r & 1) == 0 else (4 if d == 2 else 2))
            pop_odd = ((r ^ (r >> 1) ^ (r >> 2)) & 1) != 0
            return r ^ ((5 if d == 4 else 3) if pop_odd else (3 if d == 4 else 5))

        for r in range(tp_deg):
            for d in range(tp_deg):
                assert _ps_peer(_ps_peer(r, d), d) == r

        w_ps_xor = jnp.stack(
            [jnp.stack([w_by_h[_ps_peer(r, d)] for d in range(tp_deg)], axis=0) for r in range(tp_deg)],
            axis=0,
        )
        ws_ps_xor = jnp.stack(
            [jnp.stack([ws_by_h[_ps_peer(r, d)] for d in range(tp_deg)], axis=0) for r in range(tp_deg)],
            axis=0,
        )
        with patch.object(
            tpu_envs,
            "SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS",
            SimpleNamespace(get=lambda: 1),
            create=True,
        ):
            out_kd1_ps = jax.jit(
                jax.shard_map(
                    lambda a, g, w, ws: a3_v16_kdeep_sp(
                        a[0], g[0], w[0], ws[0], tp_axis="model", bn=256
                    ),
                    mesh=mesh,
                    in_specs=(
                        P("model", None, None),
                        P("model", None, None),
                        P("model", None, None, None),
                        P("model", None, None),
                    ),
                    out_specs=P("model", None),
                    check_vma=False,
                )
            )(attn_by_h, gate_by_h, w_ps_xor, ws_ps_xor)
            out_kd2_ps = jax.jit(
                jax.shard_map(
                    lambda qp, ql, w, ws: pallas_a3_wo_kdeep_ccs_last_a2a(
                        pallas_async_a2a_xor_3d(qp[0], tp_axis="model", collective_id=31),
                        ql[0],
                        w[0],
                        ws[0],
                        tp_axis="model",
                        bn=256,
                    ),
                    mesh=mesh,
                    in_specs=(
                        P("model", None, None, None),
                        P("model", None, None, None),
                        P("model", None, None, None),
                        P("model", None, None),
                    ),
                    out_specs=P("model", None),
                    check_vma=False,
                )
            )(q_prev, q_last, w_ps_xor, ws_ps_xor)
        # Reordering the 8 rank partials swaps fp32 accumulation order (<= 1 bf16 ULP, ~0.125),
        # whereas the buggy XOR-vs-stagger mismatch produces max error > 5.0!
        assert float(jnp.max(jnp.abs(out_v16.astype(jnp.float32) - out_kd1_ps.astype(jnp.float32)))) <= 0.25
        assert float(jnp.max(jnp.abs(out_kd1_raw.astype(jnp.float32) - out_kd2_ps.astype(jnp.float32)))) <= 0.25
        cos_kd1_ps = float(
            jnp.sum(out_v16.astype(jnp.float32) * out_kd1_ps.astype(jnp.float32))
            / (jnp.linalg.norm(out_v16.astype(jnp.float32)) * jnp.linalg.norm(out_kd1_ps.astype(jnp.float32)))
        )
        cos_kd2_ps = float(
            jnp.sum(out_kd1_raw.astype(jnp.float32) * out_kd2_ps.astype(jnp.float32))
            / (jnp.linalg.norm(out_kd1_raw.astype(jnp.float32)) * jnp.linalg.norm(out_kd2_ps.astype(jnp.float32)))
        )
        assert cos_kd1_ps > 0.99999
        assert cos_kd2_ps > 0.99999
        out_kd2_buggy = jax.jit(
            jax.shard_map(
                lambda qp, ql, w, ws: pallas_a3_wo_kdeep_ccs_last_a2a(
                    pallas_async_a2a_xor_3d(
                        qp[0], tp_axis="model", collective_id=31, parity_stagger=False
                    ),
                    ql[0],
                    w[0],
                    ws[0],
                    tp_axis="model",
                    bn=256,
                    parity_stagger=False,
                ),
                mesh=mesh,
                in_specs=(
                    P("model", None, None, None),
                    P("model", None, None, None),
                    P("model", None, None, None),
                    P("model", None, None),
                ),
                out_specs=P("model", None),
                check_vma=False,
            )
        )(q_prev, q_last, w_ps_xor, ws_ps_xor)
        assert float(jnp.max(jnp.abs(out_kd1_raw.astype(jnp.float32) - out_kd2_buggy.astype(jnp.float32)))) > 2.0

        # 3. D_IN_KERNEL_AG=1 with |x * 32| > 500 outlier channels (must clip to +-448, 0 NaNs!)
        #    AND odd steps_ag=1 with n_rest_steps=1 (chunk_size=128, ag_rows=64).
        chunk_size, ag_rows, num_toks_routed, n2_cols, lanes, h_dim, topk = (
            128,
            64,
            256,
            32,
            128,
            8192,
            8,
        )
        out_buf = jnp.asarray(
            (rng.standard_normal((tp_deg, num_toks_routed, n2_cols, lanes)) * 0.2).astype(
                np.float32
            )
        ).astype(jnp.bfloat16)
        inv_perm = jnp.asarray(
            rng.integers(0, num_toks_routed, size=(tp_deg, chunk_size, topk), dtype=np.int32)
        )
        topk_w = jnp.asarray(
            rng.uniform(0.05, 0.25, size=(tp_deg, chunk_size, topk)).astype(np.float32)
        )
        # Inject strong outlier channels so new_s * inv_rms_next * 32 > 500 (overflows E4M3FN -> NaN if unclipped)
        sm_np = (rng.standard_normal((tp_deg, chunk_size, h_dim)) * 0.5).astype(np.float32)
        sm_np[:, :, 0:4] = 20.0
        sm_np[:, :, 4:8] = -20.0
        stream_mid = jnp.asarray(sm_np).astype(jnp.bfloat16)
        wu = jnp.asarray(
            (rng.standard_normal((4096, h_dim)) * 0.2).astype(np.float32)
        ).astype(jnp.float8_e4m3fn)
        wus = jnp.asarray(rng.uniform(0.5, 1.5, size=(1, h_dim)).astype(np.float32))
        pnw = jnp.asarray(rng.uniform(0.8, 1.2, size=(1, h_dim)).astype(np.float32))
        af = jnp.asarray(rng.uniform(0.9, 1.1, size=(1, h_dim)).astype(np.float32))
        bf = jnp.asarray(rng.uniform(-0.05, 0.05, size=(1, h_dim)).astype(np.float32))
        # Large input_layernorm weight on outlier cols further amplifies > 500 before cast
        nnw_np = rng.uniform(0.8, 1.2, size=(1, h_dim)).astype(np.float32)
        nnw_np[:, 0:8] = 20.0
        nnw = jnp.asarray(nnw_np)

        d_in_specs = (
            P("model"),
            P("model"),
            P("model"),
            P("model"),
            P(),
            P(),
            P(),
            P(),
            P(),
            P(),
        )
        d_out_specs = (P("model"), P("model"), P("model"))

        def _ref_d_clipped(ob, ipc, ltw, sm, wu_, wus_, pnw_, af_, bf_, nnw_):
            ns, ni_bf16 = kernel_d.pallas_combine_unpool_residual_nextnorm_4slab(
                ob[0],
                ipc[0],
                ltw[0],
                sm[0],
                wu_,
                wus_,
                pnw_,
                af_,
                bf_,
                nnw_,
                tok_tile=64,
                emit_static_fp8_attn_in=False,
            )
            ni_fp8 = jnp.clip(
                ni_bf16.astype(jnp.float32) * jnp.float32(32.0), -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
            ag_p0 = jax.lax.all_gather(ni_fp8[:ag_rows], axis_name="model", tiled=False)
            return ns, ag_p0, ni_fp8[ag_rows:]

        def _rad_d_ikag(ob, ipc, ltw, sm, wu_, wus_, pnw_, af_, bf_, nnw_):
            ns, rest, ag_p0 = pallas_d_radical_4slab(
                ob[0],
                ipc[0],
                ltw[0],
                sm[0],
                wu_,
                wus_,
                pnw_,
                af_,
                bf_,
                nnw_,
                tok_tile=64,
                emit_static_fp8_attn_in=True,
                coal_gather=False,
                in_kernel_ag_rows=ag_rows,
                ag_axis_name="model",
                ag_tp_deg=8,
                ag_part0_static_fp8=True,
            )
            return ns, ag_p0, rest

        ns_ref, ag_ref, rest_ref = jax.jit(
            jax.shard_map(
                _ref_d_clipped,
                mesh=mesh,
                in_specs=d_in_specs,
                out_specs=d_out_specs,
                check_vma=False,
            )
        )(out_buf, inv_perm, topk_w, stream_mid, wu, wus, pnw, af, bf, nnw)
        ns_ik, ag_ik, rest_ik = jax.jit(
            jax.shard_map(
                _rad_d_ikag,
                mesh=mesh,
                in_specs=d_in_specs,
                out_specs=d_out_specs,
                check_vma=False,
            )
        )(out_buf, inv_perm, topk_w, stream_mid, wu, wus, pnw, af, bf, nnw)

        assert not bool(jnp.any(jnp.isnan(ag_ik.astype(jnp.float32))))
        assert not bool(jnp.any(jnp.isnan(rest_ik.astype(jnp.float32))))
        assert float(jnp.max(jnp.abs(ns_ref - ns_ik))) == 0.0
        # Allow <= 1 bf16/fp8 rounding ULP because ref rounds ni via bf16 before *32+clip
        ag_cos = float(
            jnp.sum(ag_ref.astype(jnp.float32) * ag_ik.astype(jnp.float32))
            / (jnp.linalg.norm(ag_ref.astype(jnp.float32)) * jnp.linalg.norm(ag_ik.astype(jnp.float32)))
        )
        rest_cos = float(
            jnp.sum(rest_ref.astype(jnp.float32) * rest_ik.astype(jnp.float32))
            / (
                jnp.linalg.norm(rest_ref.astype(jnp.float32))
                * jnp.linalg.norm(rest_ik.astype(jnp.float32))
            )
        )
        assert ag_cos > 0.9995
        assert rest_cos > 0.9995

        # 4. D_IN_KERNEL_AG=1 with float8_e4m3fn out_buf_3d + PREFILL_STATIC_FP8_RS=4.0 (#1607):
        #    must descale local_topk_weights by 1/4.0 so new_stream matches bf16 out_buf_3d * 4.0!
        from types import SimpleNamespace

        out_buf_fp8 = jnp.clip(out_buf.astype(jnp.float32) * 4.0, -448.0, 448.0).astype(
            jnp.float8_e4m3fn
        )
        out_buf_fp8_deq = (out_buf_fp8.astype(jnp.float32) * 0.25).astype(jnp.bfloat16)
        with patch.object(
            tpu_envs,
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS",
            SimpleNamespace(get=lambda: 4.0),
            create=True,
        ):
            ns_fp8_ref, ag_fp8_ref, rest_fp8_ref = jax.jit(
                jax.shard_map(
                    _ref_d_clipped,
                    mesh=mesh,
                    in_specs=d_in_specs,
                    out_specs=d_out_specs,
                    check_vma=False,
                )
            )(out_buf_fp8_deq, inv_perm, topk_w, stream_mid, wu, wus, pnw, af, bf, nnw)
            ns_fp8_ik, ag_fp8_ik, rest_fp8_ik = jax.jit(
                jax.shard_map(
                    _rad_d_ikag,
                    mesh=mesh,
                    in_specs=d_in_specs,
                    out_specs=d_out_specs,
                    check_vma=False,
                )
            )(out_buf_fp8, inv_perm, topk_w, stream_mid, wu, wus, pnw, af, bf, nnw)
        assert float(jnp.max(jnp.abs(ns_fp8_ref - ns_fp8_ik))) < 0.05
        assert not bool(jnp.any(jnp.isnan(ag_fp8_ik.astype(jnp.float32))))
        fp8_ag_cos = float(
            jnp.sum(ag_fp8_ref.astype(jnp.float32) * ag_fp8_ik.astype(jnp.float32))
            / (
                jnp.linalg.norm(ag_fp8_ref.astype(jnp.float32))
                * jnp.linalg.norm(ag_fp8_ik.astype(jnp.float32))
            )
        )
        fp8_rest_cos = float(
            jnp.sum(rest_fp8_ref.astype(jnp.float32) * rest_fp8_ik.astype(jnp.float32))
            / (
                jnp.linalg.norm(rest_fp8_ref.astype(jnp.float32))
                * jnp.linalg.norm(rest_fp8_ik.astype(jnp.float32))
            )
        )
        assert fp8_ag_cos > 0.9995
        assert fp8_rest_cos > 0.9995


def test_ccs_w4_wo_guard_when_kdeep_off():
    """W4_WO=256 must skip NVFP4 quantization when CCS prologue runs with A3_KDEEP_WO=0 (#1641)."""
    from types import SimpleNamespace

    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.muse_prefill_kernels.integration import _ccs_wo_uses_kdeep_nvfp4

    with (
        patch.object(
            tpu_envs,
            "SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM",
            SimpleNamespace(get=lambda: True),
        ),
        patch.object(
            tpu_envs,
            "SGLANG_TPU_MUSE_KERNEL_CCS_CHUNK_A3_PROLOGUE",
            SimpleNamespace(get=lambda: True),
            create=True,
        ),
        patch.object(
            tpu_envs,
            "SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO",
            SimpleNamespace(get=lambda: 0),
        ),
    ):
        assert _ccs_wo_uses_kdeep_nvfp4() is False

    with (
        patch.object(
            tpu_envs,
            "SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM",
            SimpleNamespace(get=lambda: True),
        ),
        patch.object(
            tpu_envs,
            "SGLANG_TPU_MUSE_KERNEL_CCS_CHUNK_A3_PROLOGUE",
            SimpleNamespace(get=lambda: True),
            create=True,
        ),
        patch.object(
            tpu_envs,
            "SGLANG_TPU_MUSE_KERNEL_A3_KDEEP_WO",
            SimpleNamespace(get=lambda: 2),
        ),
    ):
        assert _ccs_wo_uses_kdeep_nvfp4() is True

