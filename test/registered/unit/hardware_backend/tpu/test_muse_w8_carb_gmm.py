# SPDX-License-Identifier: Apache-2.0
"""Unit, bit-exact numerical, and Mosaic TPU7x HLO export tests for Wave-8 carb-k-gmm-v2."""

from __future__ import annotations

import ast
import os
import pathlib
from unittest.mock import patch

import numpy as np
import pytest

from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
    plan_prefill_chipmate_2slot_numpy_ref,
    resolve_static_fp8_rs_scale,
    scaled_fp8_1pop_strip_numpy_ref,
    simulate_e4m3fn_numpy,
)


def test_w8_carb_gmm_flags_default_off():
    """All Wave-8 carb-k-gmm-v2 flags must default strictly off / baseline."""
    pytest.importorskip("orjson")
    from sglang_tpu.envs import envs as tpu_envs

    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS.get() == 0.0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS.get() == 1
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW.get() == 0


def test_no_bare_envfield_bool_evaluation_on_w8_carb_gmm_files():
    """Ensure all modified Wave-8 GMM files call .get() on every EnvField."""
    repo = pathlib.Path(__file__).resolve().parents[5]
    rel_paths = (
        "sglang_tpu/moe/pallas/fused_rs_muse/prefill_gmm_opts.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_fused_rs_nodedup.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_v2_gather_scatter.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/fused_moe.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/kernel_d.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/pipeline.py",
    )
    checked_flags = {
        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8",
        "SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS",
        "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW",
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


def test_scaled_fp8_1pop_and_static_fp8_rs_numerics():
    """Verify 1-pop Scaled-FP8 strip GEMM and unit-RMS Static-FP8 RS numerical fidelity."""
    assert resolve_static_fp8_rs_scale(0.0) == (0.0, False)
    assert resolve_static_fp8_rs_scale(4.0) == (4.0, True)
    assert resolve_static_fp8_rs_scale(-4.0) == (4.0, False)
    with pytest.raises(ValueError, match="exceeds the guaranteed zero-clipping bound"):
        resolve_static_fp8_rs_scale(8.0)

    rng = np.random.default_rng(20261121)
    m_rows, k_dim, n_cols = 256, 4096, 256
    e2m1_mag = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    w_fp4 = (
        rng.choice([-1.0, 1.0], size=(k_dim, n_cols)).astype(np.float32)
        * e2m1_mag[rng.integers(0, 8, size=(k_dim, n_cols))]
    )
    w_sc_256 = (
        np.exp(rng.normal(-3.2, 0.35, size=(1, n_cols)))
        * np.exp(rng.normal(0.0, 0.35, size=(16, n_cols)))
    ).astype(np.float32)

    x_raw = rng.standard_normal((m_rows, k_dim)).astype(np.float32)
    x_amax = np.max(np.abs(x_raw), axis=1, keepdims=True)
    x_sc = x_amax / 448.0
    x_fp8 = simulate_e4m3fn_numpy(x_raw / x_sc)

    y_ref = np.zeros((m_rows, n_cols), dtype=np.float32)
    for b in range(16):
        y_ref += (
            x_fp8[:, b * 256 : (b + 1) * 256] @ w_fp4[b * 256 : (b + 1) * 256, :]
        ) * w_sc_256[b : b + 1, :]
    y_ref *= x_sc

    y_1pop = scaled_fp8_1pop_strip_numpy_ref(x_fp8, x_sc, w_fp4, w_sc_256, rhs_qbs=256)
    cos_sim = float(
        np.sum(y_ref * y_1pop) / (np.linalg.norm(y_ref) * np.linalg.norm(y_1pop))
    )
    rel_l2 = float(np.linalg.norm(y_ref - y_1pop) / np.linalg.norm(y_ref))
    assert cos_sim > 0.9995, f"cos_sim={cos_sim}"
    assert rel_l2 < 0.035, f"rel_l2={rel_l2}"

    # Verify k_chunk=256 chunked FP32 accumulation matches full-K Scaled-FP8 dot.
    col_max = np.max(np.abs(w_sc_256), axis=0, keepdims=True)
    col_c = col_max * (6.0 / 448.0)
    w_s = simulate_e4m3fn_numpy(w_fp4 * np.repeat(w_sc_256 / col_c, 256, axis=0))
    y_chunked = np.zeros((m_rows, n_cols), dtype=np.float32)
    for rk0 in range(0, k_dim, 256):
        y_chunked += (
            x_fp8[:, rk0 : rk0 + 256].astype(np.float32)
            @ w_s[rk0 : rk0 + 256, :].astype(np.float32)
        )
    y_chunked *= x_sc * col_c
    np.testing.assert_allclose(y_chunked, y_1pop, rtol=1e-5, atol=1e-5)

    # Verify rhs_qbs=512 Scaled-FP8 equivalence (groups_per_chunk < 1 guard).
    w_sc_512 = w_sc_256[::2, :]
    y_ref_512 = np.zeros((m_rows, n_cols), dtype=np.float32)
    for b in range(k_dim // 512):
        y_ref_512 += (
            x_fp8[:, b * 512 : (b + 1) * 512] @ w_fp4[b * 512 : (b + 1) * 512, :]
        ) * w_sc_512[b : b + 1, :]
    y_ref_512 *= x_sc
    y_512 = scaled_fp8_1pop_strip_numpy_ref(x_fp8, x_sc, w_fp4, w_sc_512, rhs_qbs=512)
    cos_512 = float(
        np.sum(y_ref_512 * y_512) / (np.linalg.norm(y_ref_512) * np.linalg.norm(y_512))
    )
    assert cos_512 > 0.9995, f"cos_512={cos_512}"

    # Unit-RMS post_expert_norm output with extreme 1-hot spike on row 0:
    z = rng.standard_normal((128, 4096)).astype(np.float32)
    z[0, :] = 0.0
    z[0, 17] = 1000.0  # Dirac spike -> exact max magnitude sqrt(4096) = 64.0
    unit_rms = z / np.sqrt(np.mean(z * z, axis=1, keepdims=True) + 1e-8)
    assert np.max(np.abs(unit_rms)) <= 64.0001
    q_rs = simulate_e4m3fn_numpy(unit_rms * 4.0) * 0.25
    assert np.isclose(q_rs[0, 17], 64.0, rtol=0.0, atol=1e-5)
    cos_rs = float(
        np.sum(unit_rms * q_rs) / (np.linalg.norm(unit_rms) * np.linalg.norm(q_rs))
    )
    assert cos_rs > 0.9995


def test_chipmate_2slot_planner_reduces_skew_and_matches_jax():
    """Verify 2-slot chipmate strictly improves multi-expert skew and matches JAX 0-ULP."""
    ep_size = 8
    e_loc = 32
    tile_m = 256
    rng = np.random.default_rng(777)

    # Construct a skewed batch where rank 0 has TWO hot experts (each 6 tiles = 1536 rows)
    # while its chipmate rank 1 is lightly loaded: 1-slot can only split 1 expert, whereas
    # 2-slot rebalances both hot experts!
    ids = rng.integers(0, ep_size * e_loc, size=(4096, 8), dtype=np.int32)
    # Clear rank 1 and overload rank 0 slots 3 and 11:
    ids = np.where((ids >= e_loc) & (ids < 2 * e_loc), 2 * e_loc + (ids % e_loc), ids)
    flat = ids.reshape(-1)
    flat[:1536] = 3
    flat[1536:3072] = 11
    ids[-64:, :] = ep_size * e_loc  # pad sentinels

    v_np, cm_np = plan_prefill_chipmate_2slot_numpy_ref(
        ids, ep_size=ep_size, num_local_experts=e_loc, tile_m=tile_m, mode="split"
    )
    # Both borrowed slots on pair (0, 1) must fire on distinct experts {3, 11}!
    assert cm_np[0, 0] in (3, 11) and cm_np[0, 3] in (3, 11)
    assert cm_np[0, 0] != cm_np[0, 3]
    assert cm_np[1, 1] == 1 and cm_np[1, 4] == 1
    assert np.all(v_np[-64:, :] == ep_size * (e_loc + 2))
    # Conservation of valid non-sentinel assignments:
    assert int(np.sum(v_np < ep_size * (e_loc + 2))) == int(np.sum(ids < ep_size * e_loc))

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
        plan_prefill_chipmate_2slot_jax,
    )

    for mode in ("whole", "split"):
        v_ref, cm_ref = plan_prefill_chipmate_2slot_numpy_ref(
            ids, ep_size=ep_size, num_local_experts=e_loc, tile_m=tile_m, mode=mode
        )
        v_jax, cm_jax = plan_prefill_chipmate_2slot_jax(
            jnp.asarray(ids),
            ep_size=ep_size,
            num_local_experts=e_loc,
            tile_m=tile_m,
            mode=mode,
            use_pallas=False,
        )
        np.testing.assert_array_equal(np.asarray(cm_jax), cm_ref)
        np.testing.assert_array_equal(np.asarray(v_jax), v_ref)


def test_w8_carb_gmm_tpu7x_mosaic_hlo_export_at_t16k():
    """Verify all Wave-8 carb-k-gmm-v2 modes lower cleanly to TPU7x Mosaic HLO at T=16384."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    base_env = {
        "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
        "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE": "1",
        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
        "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "1",
        "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "0",
        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
        "SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL": "4096",
        "SGLANG_TPU_MUSE_KERNEL_D_T64_C16": "1",
        "SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP": "1",
    }

    arms = (
        (
            {
                **base_env,
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "3",
                "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
            },
            ("-sw3", "-cm1d"),
        ),
        (
            {
                **base_env,
                "SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "-4.0",
            },
            ("-eo1", "-sfp8_4"),
        ),
        (
            {
                **base_env,
                "SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8": "2",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "4.0",
                "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
            },
            ("E_33-M_131072", "-eo2", "-sfp8_4", "-cm1d"),
        ),
        (
            {
                **base_env,
                "SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8": "2",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "4.0",
                "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
            },
            ("E_34-M_131072", "-eo2", "-sfp8_4", "-cm1s2d", "pallas_prefill_chipmate_2slot_split"),
        ),
        (
            {
                **base_env,
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "2",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "4.0",
                "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "3",
            },
            ("E_35-M_131072", "-sw2", "-sfp8_4", "-cm1s3d", "pallas_prefill_chipmate_3slot_split"),
        ),
        (
            {
                **base_env,
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
                "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2": "2",
                "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_K_STEP": "512",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "2",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "4.0",
                "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
            },
            ("E_34-M_131072", "-sw2", "-sfp8_4", "-cm1s2d"),
        ),
    )

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        patch("jax.default_backend", return_value="tpu"),
        tpu_info_for_cpu_export(),
    ):
        try:
            for env_dict, expected_tokens in arms:
                with patch.dict(os.environ, env_dict):
                    jax.clear_caches()

                    def _wrap(
                        a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus, vt
                    ):
                        return jax.shard_map(
                            lambda *xs: fused_moe_full_bridge_and_ffn(
                                *xs[:-1],
                                mesh=mesh,
                                router_mode="bf16_hilo",
                                b2_layout="nstream",
                                b2_two_half=True,
                                b2_deq0_first=True,
                                hoist_routing=True,
                                d_layout="4slab",
                                d_tok_tile=64,
                                d_fp8_unpool=True,
                                in_shard_map=True,
                                b2_deq0_new_buffer=True,
                                valid_tokens=xs[-1],
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
                            pfw, aa, ba, pfw, vt,
                        )

                    lowered = jax.jit(_wrap).trace(
                        jax.ShapeDtypeStruct((16384, 8192), jnp.bfloat16),
                        jax.ShapeDtypeStruct((16384, 8192), jnp.float32),
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
                        jax.ShapeDtypeStruct((1,), jnp.int32),
                    ).lower(lowering_platforms=("tpu",))
                    hlo = lowered.as_text("stablehlo")
                    assert "partition_id" not in hlo
                    for tok in expected_tokens:
                        assert tok in hlo, f"Missing {tok!r} in lowered HLO"

            # Verify expand_fp4_strip_to_scaled_fp8 handles rhs_qbs=512 (COMPACT_FP4_QBLOCK=512).
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
                expand_fp4_strip_to_scaled_fp8,
            )

            def _expand512_ker(w_ref, s_ref, out_ref, scratch_ref):
                col_c = expand_fp4_strip_to_scaled_fp8(
                    w_ref, s_ref, scratch_ref,
                    start_n=0, col_size=256, tile_k=4096,
                    rhs_qbs=512, packing=8, rhs_quant_dtype=jnp.float4_e2m1fn,
                )
                out_ref[...] = jnp.broadcast_to(col_c, (8, 256))

            lowered_512 = jax.jit(
                pl.pallas_call(
                    _expand512_ker,
                    out_shape=jax.ShapeDtypeStruct((8, 256), jnp.float32),
                    in_specs=[
                        pl.BlockSpec(memory_space=pltpu.VMEM),
                        pl.BlockSpec(memory_space=pltpu.VMEM),
                    ],
                    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                    scratch_shapes=[pltpu.VMEM((4096, 256), jnp.float8_e4m3fn)],
                )
            ).trace(
                jax.ShapeDtypeStruct((512, 256), jnp.uint32),
                jax.ShapeDtypeStruct((8, 256), jnp.bfloat16),
            ).lower(lowering_platforms=("tpu",))
            assert len(lowered_512.as_text("stablehlo")) > 0
        finally:
            jax.clear_caches()



def test_chipmate_3slot_hypercube_reduces_cross_chip_skew_and_matches_jax():
    """Verify 3-slot hypercube (1,2,4) flattens cross-chip skew and matches JAX 0-ULP."""
    ep_size = 8
    e_loc = 32
    tile_m = 256
    rng = np.random.default_rng(20261122)

    # Overload an entire 2-TC chip (ranks 0 and 1 equally heavy so xor=1 intra-chip cannot help!)
    # while ranks 2 and 4 on neighbor chips are light: 3-slot hypercube (1,2,4) must rebalance!
    ids = (rng.integers(2, ep_size, size=(4096, 8), dtype=np.int32) * e_loc + rng.integers(0, 4, size=(4096, 8), dtype=np.int32))
    flat = ids.reshape(-1)
    flat[0:12288] = 5
    flat[12288:24576] = e_loc + 7
    ids[-32:, :] = ep_size * e_loc

    v2, cm2 = plan_prefill_chipmate_2slot_numpy_ref(
        ids, ep_size=ep_size, num_local_experts=e_loc, tile_m=tile_m,
        mode="split", slot_xors=(1, 1),
    )
    v3, cm3 = plan_prefill_chipmate_2slot_numpy_ref(
        ids, ep_size=ep_size, num_local_experts=e_loc, tile_m=tile_m,
        mode="split", slot_xors=(1, 2, 4),
    )
    assert cm3.shape == (ep_size, 9)
    # Ranks 0/1 must offload via inter-chip rounds (rnd=1 xor=2 and/or rnd=2 xor=4).
    assert cm3[2, 4] == 1 or cm3[4, 7] == 1
    assert int(np.sum(v3 < ep_size * (e_loc + 3))) == int(np.sum(ids < ep_size * e_loc))
    # Max tiles across 8 ranks strictly lower with 3-slot hypercube vs 2-slot intra-chip.
    c2 = np.bincount(v2[v2 < ep_size * (e_loc + 2)], minlength=ep_size * (e_loc + 2)).reshape(ep_size, -1)
    c3 = np.bincount(v3[v3 < ep_size * (e_loc + 3)], minlength=ep_size * (e_loc + 3)).reshape(ep_size, -1)
    max_t2 = int(np.max(np.sum((c2 + tile_m - 1) // tile_m, axis=1)))
    max_t3 = int(np.max(np.sum((c3 + tile_m - 1) // tile_m, axis=1)))
    assert max_t3 < max_t2, f"Expected hypercube max_t3={max_t3} < max_t2={max_t2}"

    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
        plan_prefill_chipmate_2slot_jax,
    )

    v3_jax, cm3_jax = plan_prefill_chipmate_2slot_jax(
        jnp.asarray(ids), ep_size=ep_size, num_local_experts=e_loc,
        tile_m=tile_m, mode="split", use_pallas=False, slot_xors=(1, 2, 4),
    )
    np.testing.assert_array_equal(np.asarray(cm3_jax), cm3)
    np.testing.assert_array_equal(np.asarray(v3_jax), v3)


def test_flq_zero_rmw_and_gmm1_rowmax_fusion_interpret_parity():
    """Verify GMM_FLQ_ZERO_RMW=2 matches baseline 0-ULP under Pallas interpret=True."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_v2_gather_scatter import (
        GmmConfigs,
        Dimensions,
        TileSizes,
        InputConfigs,
        MetadataRef,
        WeightsRef,
        inner_kernel,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    rng = np.random.default_rng(99)
    m_rows, k1, n1_half, k2, n2 = 256, 512, 256, 256, 256
    # Use a 2-strip K=512, N=512 (256 gate + 256 up) GMM1 followed by GMM2 to exercise both
    # 32-row flq and fused GMM1->GMM2 rowmax reuse.
    lhs_3d = jax.device_put(
        jnp.asarray(rng.standard_normal((m_rows, k1 // 128, 128)).astype(np.float32), jnp.bfloat16)
    )
    w1_u32 = jnp.asarray(rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32))
    s1_bf16 = jnp.asarray(np.exp(rng.normal(-2.0, 0.2, size=(k1 // 256, n1_half * 2))).astype(np.float32), jnp.bfloat16)
    w2_u32 = jnp.asarray(rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32))
    s2_bf16 = jnp.asarray(np.exp(rng.normal(-2.0, 0.2, size=(k2 // 256, n2))).astype(np.float32), jnp.bfloat16)

    cfgs1 = GmmConfigs(
        dims=Dimensions(size_m=8192, size_k=k1, size_n=n1_half * 2, size_group=1, size_lhs_group=1, size_lhs_sublane=8),
        tiles=TileSizes(tile_m=m_rows, tile_k=k1, tile_n=n1_half * 2),
        lhs_cfgs=InputConfigs(quant_dtype=jnp.float8_e4m3fn.dtype, quant_block_size=k1, dtype=jnp.bfloat16),
        rhs_cfgs=InputConfigs(quant_dtype=jnp.float4_e2m1fn.dtype, quant_block_size=256, dtype=jnp.float4_e2m1fn, has_bias=False, has_scale=True, packing=8, num_quant_blocks=k1 // 256),
        out_dtype=jnp.bfloat16, acc_dtype=jnp.float32, zero_init=False, fuse_act="silu",
    )
    cfgs2 = GmmConfigs(
        dims=Dimensions(size_m=8192, size_k=k2, size_n=n2, size_group=1, size_lhs_group=1, size_lhs_sublane=8),
        tiles=TileSizes(tile_m=m_rows, tile_k=k2, tile_n=n2),
        lhs_cfgs=InputConfigs(quant_dtype=jnp.float8_e4m3fn.dtype, quant_block_size=k2, dtype=jnp.bfloat16),
        rhs_cfgs=InputConfigs(quant_dtype=jnp.float4_e2m1fn.dtype, quant_block_size=256, dtype=jnp.float4_e2m1fn, has_bias=False, has_scale=True, packing=8, num_quant_blocks=k2 // 256),
        out_dtype=jnp.bfloat16, acc_dtype=jnp.float32, zero_init=False, fuse_act=None,
    )

    def _run_pair(flq_zero_rmw_val: str):
        def _ker(x3d_r, w1_r, s1_r, w2_r, s2_r, out_r, g1_r, pq_fp8_r, pq_sc_r, p1_r, acc_r, m_gid, m_off):
            m_gid[0] = jnp.int32(0)
            m_off[0] = jnp.int32(0)
            m_off[1] = jnp.int32(m_rows)
            meta = MetadataRef(gm_id_to_group_id=m_gid, gm_id_to_m_offset=m_off)
            inner_kernel(
                None, WeightsRef(weight=w1_r, scale=s1_r, bias=None), g1_r, p1_r, acc_r, meta,
                cfgs=cfgs1, scatter_mode=True, _k_id=0, _num_k=1, _gm_id=0, _n_id=0, _num_n=1,
                fused_g2_amax_slot=1, prequant_lhs_fp8_ref=pq_fp8_r, prequant_lhs_scale_ref=pq_sc_r,
                skip_row_mask=True, lhs_3d_vmem_ref=x3d_r,
            )
            inner_kernel(
                None, WeightsRef(weight=w2_r, scale=s2_r, bias=None), out_r, p1_r, acc_r, meta,
                cfgs=cfgs2, scatter_mode=True, _k_id=0, _num_k=1, _gm_id=0, _n_id=0, _num_n=1,
                fused_g2_amax_slot=1, prequant_lhs_fp8_ref=pq_fp8_r.at[:, :k2], prequant_lhs_scale_ref=pq_sc_r,
                skip_row_mask=True, lhs_2d_vmem_ref=g1_r,
            )

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
            patch.dict(os.environ, {
                "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": flq_zero_rmw_val,
            }),
        ):
            call = pl.pallas_call(
                _ker,
                out_shape=jax.ShapeDtypeStruct((m_rows, n2), jnp.bfloat16),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 5,
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((m_rows, n1_half), jnp.bfloat16),
                    pltpu.VMEM((m_rows, k1), jnp.float8_e4m3fn),
                    pltpu.VMEM((2, m_rows, 128), jnp.float32),
                    pltpu.VMEM((8, n1_half * 2), jnp.float32),
                    pltpu.VMEM((m_rows, n1_half * 2), jnp.float32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                ],
                interpret=True,
            )
            return np.asarray(call(lhs_3d, w1_u32, s1_bf16, w2_u32, s2_bf16), dtype=np.float32)

    y_base = _run_pair("0")
    y_opt = _run_pair("2")
    assert np.all(np.isfinite(y_opt))
    np.testing.assert_array_equal(y_opt, y_base)


def test_c24k_smem_window_no_spill_and_mosaic_lowering():
    """Verify M=196608 (T=24k) stays < 512 KiB SMEM without u16x2 and lowers cleanly to Mosaic."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh

    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        fused_rs_smem_rows,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    smem_rows = fused_rs_smem_rows(196_608, 8, False)
    assert smem_rows == 98_304
    num_slots = 34 + (smem_rows + 255) // 256 - 1
    smem_bytes = (
        smem_rows * 4  # plain int32 primary_idx_ref (no u16x2 overhead)
        + (num_slots + 1) * 26 * 4  # flattened 1D block_scatter secondary_idx_ref
        + (2 * num_slots + 1) * 4  # metadata_ref
        + (5 + 2 * num_slots) * 4  # group_start
    )
    assert smem_bytes < 512 * 1024, f"M=196608 SMEM exceeded 512 KiB: {smem_bytes}"

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    env_dict = {
        "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
        "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE": "1",
        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
        "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "0",
        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
        "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "4.0",
        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
        "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
        "SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL": "4096",
        "SGLANG_TPU_MUSE_KERNEL_D_T64_C16": "1",
        "SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP": "1",
    }
    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        patch("jax.default_backend", return_value="tpu"),
        tpu_info_for_cpu_export(),
        patch.dict(os.environ, env_dict),
    ):
        jax.clear_caches()

        def _wrap(
            a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus,
            npnw, naa, nba, npfw, vt,
        ):
            return jax.shard_map(
                lambda *xs: fused_moe_full_bridge_and_ffn(
                    *xs[:-1],
                    mesh=mesh,
                    router_mode="bf16_hilo",
                    b2_layout="nstream",
                    b2_two_half=True,
                    b2_deq0_first=True,
                    hoist_routing=True,
                    d_layout="4slab",
                    d_tok_tile=64,
                    d_fp8_unpool=True,
                    in_shard_map=True,
                    b2_deq0_new_buffer=True,
                    valid_tokens=xs[-1],
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
                    jax.sharding.PartitionSpec("model", None, None),
                    jax.sharding.PartitionSpec("model", None, None),
                    jax.sharding.PartitionSpec("model", None, None),
                    jax.sharding.PartitionSpec("model", None, None),
                    jax.sharding.PartitionSpec(),
                    jax.sharding.PartitionSpec("model"),
                    jax.sharding.PartitionSpec(),
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
                npnw, naa, nba, npfw, vt,
            )

        sd = jax.ShapeDtypeStruct
        hlo = (
            jax.jit(_wrap)
            .trace(
                sd((24576, 8192), jnp.bfloat16),
                sd((24576, 8192), jnp.float32),
                sd((8192,), jnp.float32),
                sd((8192,), jnp.float32),
                sd((8192,), jnp.bfloat16),
                sd((8192, 256), jnp.float32),
                sd((256,), jnp.float32),
                sd((8192, 4096), jnp.float8_e4m3fn),
                sd((4096,), jnp.float32),
                sd((4096,), jnp.bfloat16),
                sd((256, 4096, 8192), jnp.float4_e2m1fn),
                sd((256, 4096, 4096), jnp.float4_e2m1fn),
                sd((256, 8, 8192), jnp.bfloat16),
                sd((256, 8, 4096), jnp.bfloat16),
                sd((4096,), jnp.bfloat16),
                sd((8,), jnp.int32),
                sd((4096, 8192), jnp.float8_e4m3fn),
                sd((8192,), jnp.float32),
                sd((8192,), jnp.bfloat16),
                sd((8192,), jnp.float32),
                sd((8192,), jnp.float32),
                sd((8192,), jnp.bfloat16),
                sd((1,), jnp.int32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert "E_34-M_196608" in hlo
        assert "-fbs-" in hlo
        assert "-u16w-" not in hlo


def test_rb4_and_flq2_fp8_first_0ulp_interpret_parity():
    """Verify rb4 (128-row bucket inside 256-row VMEM) and flq2_fp8_first are 0-ULP vs 256-row."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_v2_gather_scatter import (
        Dimensions,
        GmmConfigs,
        InputConfigs,
        MetadataRef,
        TileSizes,
        WeightsRef,
        inner_kernel,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    rng = np.random.default_rng(20261122)
    tile_m, k1, n1_half, k2, n2, valid_rows = 256, 512, 256, 256, 256, 96
    lhs_3d = jax.device_put(
        jnp.asarray(rng.standard_normal((tile_m, k1 // 128, 128)).astype(np.float32), jnp.bfloat16)
    )
    w1_u32 = jnp.asarray(rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32))
    s1_bf16 = jnp.asarray(np.exp(rng.normal(-2.0, 0.2, size=(k1 // 256, n1_half * 2))).astype(np.float32), jnp.bfloat16)
    w2_u32 = jnp.asarray(rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32))
    s2_bf16 = jnp.asarray(np.exp(rng.normal(-2.0, 0.2, size=(k2 // 256, n2))).astype(np.float32), jnp.bfloat16)

    def _run_bucket(m_rows: int, flq_val: str) -> np.ndarray:
        cfgs1 = GmmConfigs(
            dims=Dimensions(size_m=16384, size_k=k1, size_n=n1_half * 2, size_group=1, size_lhs_group=1, size_lhs_sublane=8),
            tiles=TileSizes(tile_m=m_rows, tile_k=k1, tile_n=n1_half * 2),
            lhs_cfgs=InputConfigs(quant_dtype=jnp.float8_e4m3fn.dtype, quant_block_size=k1, dtype=jnp.bfloat16),
            rhs_cfgs=InputConfigs(quant_dtype=jnp.float4_e2m1fn.dtype, quant_block_size=256, dtype=jnp.float4_e2m1fn, has_bias=False, has_scale=True, packing=8, num_quant_blocks=k1 // 256),
            out_dtype=jnp.bfloat16, acc_dtype=jnp.float32, zero_init=False, fuse_act="silu",
        )
        cfgs2 = GmmConfigs(
            dims=Dimensions(size_m=16384, size_k=k2, size_n=n2, size_group=1, size_lhs_group=1, size_lhs_sublane=8),
            tiles=TileSizes(tile_m=m_rows, tile_k=k2, tile_n=n2),
            lhs_cfgs=InputConfigs(quant_dtype=jnp.float8_e4m3fn.dtype, quant_block_size=k2, dtype=jnp.bfloat16),
            rhs_cfgs=InputConfigs(quant_dtype=jnp.float4_e2m1fn.dtype, quant_block_size=256, dtype=jnp.float4_e2m1fn, has_bias=False, has_scale=True, packing=8, num_quant_blocks=k2 // 256),
            out_dtype=jnp.bfloat16, acc_dtype=jnp.float32, zero_init=False, fuse_act=None,
        )

        def _ker(x3d_r, w1_r, s1_r, w2_r, s2_r, out_r, g1_r, pq_fp8_r, pq_sc_r, p1_r, acc_r, m_gid, m_off):
            m_gid[0] = jnp.int32(0)
            m_off[0] = jnp.int32(0)
            m_off[1] = jnp.int32(valid_rows)
            meta = MetadataRef(gm_id_to_group_id=m_gid, gm_id_to_m_offset=m_off)
            inner_kernel(
                None, WeightsRef(weight=w1_r, scale=s1_r, bias=None),
                g1_r.at[pl.ds(0, m_rows), :], p1_r, acc_r.at[pl.ds(0, m_rows), :], meta,
                cfgs=cfgs1, scatter_mode=True, _k_id=0, _num_k=1, _gm_id=0, _n_id=0, _num_n=1,
                fused_g2_amax_slot=1, prequant_lhs_fp8_ref=pq_fp8_r.at[pl.ds(0, m_rows), :],
                prequant_lhs_scale_ref=pq_sc_r.at[:, pl.ds(0, m_rows), :],
                skip_row_mask=True, lhs_3d_vmem_ref=x3d_r.at[pl.ds(0, m_rows), :, :],
            )
            inner_kernel(
                None, WeightsRef(weight=w2_r, scale=s2_r, bias=None),
                out_r.at[pl.ds(0, m_rows), :], p1_r, acc_r.at[pl.ds(0, m_rows), :], meta,
                cfgs=cfgs2, scatter_mode=True, _k_id=0, _num_k=1, _gm_id=0, _n_id=0, _num_n=1,
                fused_g2_amax_slot=1, prequant_lhs_fp8_ref=pq_fp8_r.at[pl.ds(0, m_rows), :k2],
                prequant_lhs_scale_ref=pq_sc_r.at[:, pl.ds(0, m_rows), :],
                skip_row_mask=True, lhs_2d_vmem_ref=g1_r.at[pl.ds(0, m_rows), :],
            )

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
            patch.dict(os.environ, {
                "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": flq_val,
            }),
        ):
            call = pl.pallas_call(
                _ker,
                out_shape=jax.ShapeDtypeStruct((tile_m, n2), jnp.bfloat16),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 5,
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((tile_m, n1_half), jnp.bfloat16),
                    pltpu.VMEM((tile_m, k1), jnp.float8_e4m3fn),
                    pltpu.VMEM((2, tile_m, 128), jnp.float32),
                    pltpu.VMEM((8, n1_half * 2), jnp.float32),
                    pltpu.VMEM((tile_m, n1_half * 2), jnp.float32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                ],
                interpret=True,
            )
            return np.asarray(call(lhs_3d, w1_u32, s1_bf16, w2_u32, s2_bf16), dtype=np.float32)[:valid_rows]

    y_256 = _run_bucket(256, "2")
    y_128 = _run_bucket(128, "2")
    y_ref = _run_bucket(256, "0")
    assert np.all(np.isfinite(y_128))
    np.testing.assert_array_equal(y_256, y_ref)
    np.testing.assert_array_equal(y_128, y_256)


def test_rb4_and_g2a4_g2a5_mosaic_lowering():
    """Verify PREFILL_ROW_BUCKETS=4 and GMM_GATHER_2AHEAD=4/5 lower cleanly at T=24k and T=16k."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh

    import sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline  # noqa: F401
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_moe import moe_gmm_local_rs_nodedup
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    base_env = {
        "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
        "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
        "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
        "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "2",
        "SGLANG_TPU_MUSE_KERNEL_VECTORIZED_DEQUANT": "1",
    }

    in_specs = (
        jax.sharding.PartitionSpec(),
        jax.sharding.PartitionSpec("model"),
        jax.sharding.PartitionSpec("model"),
        jax.sharding.PartitionSpec("model"),
        jax.sharding.PartitionSpec("model"),
        jax.sharding.PartitionSpec("model"),
        jax.sharding.PartitionSpec("model", None),
        jax.sharding.PartitionSpec(),
        jax.sharding.PartitionSpec(),
    )

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        patch("jax.default_backend", return_value="tpu"),
        tpu_info_for_cpu_export(),
    ):
        try:
            for g2a_mode in ("4", "5"):
                for T in (24576, 16384):
                    with patch.dict(os.environ, {**base_env, "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": g2a_mode}):
                        jax.clear_caches()
                        sd = jax.ShapeDtypeStruct
                        hlo = (
                            jax.jit(
                                lambda *xs: jax.shard_map(
                                    lambda hs, w1, w1s, w2, w2s, go, wts, ids, pnw: moe_gmm_local_rs_nodedup(
                                        hs, w1, w1s, None, w2, w2s, None, None, None, go, wts, ids, pnw,
                                        activation="silu", topk=8, ep_size=8, ep_axis_name="model",
                                        has_post_norm=True, sp_enabled=True, fp8_direct_write=False,
                                        expert_tp=False, fp4_w4a16=False, block_scatter=True, return_raw_out_buf=True,
                                    ),
                                    mesh=mesh, in_specs=in_specs,
                                    out_specs=jax.sharding.PartitionSpec("model", None, None),
                                    check_vma=False,
                                )(*xs)
                            )
                            .trace(
                                sd((T, 4096), jnp.bfloat16),
                                sd((256, 4096, 8192), jnp.float4_e2m1fn),
                                sd((256, 8, 8192), jnp.bfloat16),
                                sd((256, 4096, 4096), jnp.float4_e2m1fn),
                                sd((256, 8, 4096), jnp.bfloat16),
                                sd((8,), jnp.int32),
                                sd((T, 8), jnp.float32),
                                sd((T, 8), jnp.int32),
                                sd((4096,), jnp.bfloat16),
                            )
                            .lower(lowering_platforms=("tpu",))
                            .as_text("stablehlo")
                        )
                        assert "-rb4-" in hlo
                        assert f"-g2a{g2a_mode}-" in hlo
        finally:
            jax.clear_caches()
