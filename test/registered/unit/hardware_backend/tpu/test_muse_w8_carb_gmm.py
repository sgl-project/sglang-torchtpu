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
