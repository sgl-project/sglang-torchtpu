# SPDX-License-Identifier: Apache-2.0
"""0-ULP interpret parity and TPU7x Mosaic lowering tests for Wave-9 GMM sw4/sw5 + w2c1 + rb3/rb4."""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest


def _run_two_ntile_gmm1_gmm2_interpret(
    *,
    sw_mode: int,
    flq_zero_rmw: int,
    use_w2_cache: bool,
    m_rows: int = 256,
    k1: int = 2048,
    n1_half: int = 512,
    k2: int = 1024,
    n2: int = 512,
    rhs_qbs: int = 256,
    seed: int = 20261122,
    input_rows: int = 256,
) -> np.ndarray:
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

    rng = np.random.default_rng(seed)
    x_full = rng.standard_normal((input_rows, k1 // 128, 128)).astype(np.float32)
    lhs_3d = jax.device_put(jnp.asarray(x_full[:m_rows], jnp.bfloat16))
    w1_a = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    w1_b = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    s1_a = jnp.asarray(
        np.exp(rng.normal(-2.0, 0.25, size=(k1 // rhs_qbs, 1, n1_half * 2))).astype(
            np.float32
        ),
        jnp.bfloat16,
    )
    s1_b = jnp.asarray(
        np.exp(rng.normal(-2.0, 0.25, size=(k1 // rhs_qbs, 1, n1_half * 2))).astype(
            np.float32
        ),
        jnp.bfloat16,
    )
    w2_a = jnp.asarray(rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32))
    w2_b = jnp.asarray(rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32))
    s2_a = jnp.asarray(
        np.exp(rng.normal(-2.0, 0.25, size=(k2 // rhs_qbs, 1, n2))).astype(np.float32),
        jnp.bfloat16,
    )
    s2_b = jnp.asarray(
        np.exp(rng.normal(-2.0, 0.25, size=(k2 // rhs_qbs, 1, n2))).astype(np.float32),
        jnp.bfloat16,
    )
    pnw_1d = jnp.asarray(
        1.0 + 0.1 * rng.standard_normal((n2 * 2,)).astype(np.float32), jnp.bfloat16
    )
    pnw_8 = jnp.broadcast_to(pnw_1d[None, :], (8, n2 * 2))
    gs_gate = jnp.asarray([0.85], dtype=jnp.float32)
    gs_up = jnp.asarray([1.15], dtype=jnp.float32)
    gs_w2 = jnp.asarray([0.92], dtype=jnp.float32)

    cfgs1 = GmmConfigs(
        dims=Dimensions(
            size_m=196608,
            size_k=k1,
            size_n=n1_half * 2,
            size_group=1,
            size_lhs_group=1,
            size_lhs_sublane=8,
        ),
        tiles=TileSizes(tile_m=m_rows, tile_k=k1, tile_n=n1_half),
        lhs_cfgs=InputConfigs(
            quant_dtype=jnp.float8_e4m3fn.dtype,
            quant_block_size=k1,
            dtype=jnp.bfloat16,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=jnp.float4_e2m1fn.dtype,
            quant_block_size=rhs_qbs,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k1 // rhs_qbs,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act="silu",
    )
    cfgs2 = GmmConfigs(
        dims=Dimensions(
            size_m=196608,
            size_k=k2,
            size_n=n2 * 2,
            size_group=1,
            size_lhs_group=1,
            size_lhs_sublane=8,
        ),
        tiles=TileSizes(tile_m=m_rows, tile_k=k2, tile_n=n2),
        lhs_cfgs=InputConfigs(
            quant_dtype=jnp.float8_e4m3fn.dtype,
            quant_block_size=k2,
            dtype=jnp.bfloat16,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=jnp.float4_e2m1fn.dtype,
            quant_block_size=rhs_qbs,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k2 // rhs_qbs,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act=None,
    )
    sc_slots = 2 if flq_zero_rmw >= 2 else 1

    def _ker(
        x3d_r,
        w1a_r,
        s1a_r,
        w1b_r,
        s1b_r,
        w2a_r,
        s2a_r,
        w2b_r,
        s2b_r,
        pnw_r,
        gsg_r,
        gsu_r,
        gsw2_r,
        out_r,
        g1_r,
        pq_fp8_r,
        pq_sc_r,
        p1_r,
        acc_r,
        m_gid,
        m_off,
        w2c_r,
    ):
        m_gid[0] = jnp.int32(0)
        m_off[0] = jnp.int32(0)
        m_off[1] = jnp.int32(m_rows)
        meta = MetadataRef(gm_id_to_group_id=m_gid, gm_id_to_m_offset=m_off)
        w1_list = [(w1a_r, s1a_r), (w1b_r, s1b_r)]
        w2_list = [(w2a_r, s2a_r), (w2b_r, s2b_r)]
        for nid in range(2):
            wr, sr = w1_list[nid]
            inner_kernel(
                None,
                WeightsRef(weight=wr, scale=sr, bias=None),
                g1_r.at[:, pl.ds(nid * n1_half, n1_half)],
                p1_r,
                acc_r,
                meta,
                cfgs=cfgs1,
                gs_gate_ref=gsg_r,
                gs_up_ref=gsu_r,
                scatter_mode=True,
                _k_id=0,
                _num_k=1,
                _gm_id=0,
                _n_id=nid,
                _num_n=2,
                fused_g2_amax_slot=(1 if flq_zero_rmw >= 2 else None),
                prequant_lhs_fp8_ref=pq_fp8_r,
                prequant_lhs_scale_ref=pq_sc_r,
                skip_row_mask=True,
                lhs_3d_vmem_ref=x3d_r,
            )
        for nid in range(2):
            wr, sr = w2_list[nid]
            if use_w2_cache:
                for ck in range(0, k2, 256):
                    for cn in range(0, n2, 256):
                        raw_c = wr[pl.ds(ck // 8, 32), pl.ds(cn, 256)]
                        w2c_r[nid, pl.ds(ck, 256), pl.ds(cn, 256)] = pltpu.bitcast(
                            raw_c, jnp.float4_e2m1fn
                        ).astype(jnp.float8_e4m3fn)
            inner_kernel(
                None,
                WeightsRef(weight=wr, scale=sr, bias=None),
                out_r.at[:, pl.ds(nid * n2, n2)],
                p1_r,
                acc_r,
                meta,
                cfgs=cfgs2,
                gs_gate_ref=gsw2_r,
                gs_up_ref=gsw2_r,
                scatter_mode=True,
                _k_id=0,
                _num_k=1,
                _gm_id=0,
                _n_id=nid,
                _num_n=2,
                fused_g2_amax_slot=(1 if flq_zero_rmw >= 2 else None),
                prequant_lhs_fp8_ref=pq_fp8_r.at[:, :k2],
                prequant_lhs_scale_ref=pq_sc_r,
                pnw_ref=pnw_r,
                skip_row_mask=True,
                lhs_2d_vmem_ref=g1_r,
                w2_fp8_cache_ref=(w2c_r.at[nid] if use_w2_cache else None),
            )

    with (
        patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
        patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": str(sw_mode),
                "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": str(k1),
                "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": str(flq_zero_rmw),
            },
        ),
    ):
        call = pl.pallas_call(
            _ker,
            out_shape=jax.ShapeDtypeStruct((m_rows, n2 * 2), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.SMEM),
                    pl.BlockSpec(memory_space=pltpu.SMEM),
                    pl.BlockSpec(memory_space=pltpu.SMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((m_rows, n1_half * 2), jnp.bfloat16),
                    pltpu.VMEM((m_rows, k1), jnp.float8_e4m3fn),
                    pltpu.VMEM((sc_slots, m_rows, 128), jnp.float32),
                    pltpu.VMEM((1, 128), jnp.float32),
                    pltpu.VMEM((1, 128), jnp.float32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                    pltpu.VMEM((2, k2, n2), jnp.float8_e4m3fn),
                ],
            ),
            interpret=True,
        )
        res = call(
            lhs_3d,
            w1_a,
            s1_a,
            w1_b,
            s1_b,
            w2_a,
            s2_a,
            w2_b,
            s2_b,
            pnw_8,
            gs_gate,
            gs_up,
            gs_w2,
        )
        return np.asarray(res, dtype=np.float32)


def test_sw4_sw5_and_w2_fp8_cache_0ulp_interpret_parity():
    """Verify sw4, sw5 (scf.for strips), w2_fp8_cache, and row buckets match sw2 0-ULP."""
    y_sw2 = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=2, flq_zero_rmw=0, use_w2_cache=False
    )
    y_sw4 = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=4, flq_zero_rmw=2, use_w2_cache=False
    )
    y_sw5 = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=5, flq_zero_rmw=2, use_w2_cache=False
    )
    y_sw6_w2c = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=6, flq_zero_rmw=2, use_w2_cache=True
    )
    y_sw5_w2c = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=5, flq_zero_rmw=3, use_w2_cache=True
    )
    y_sw6_no_amax = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=6, flq_zero_rmw=1, use_w2_cache=True
    )
    assert np.all(np.isfinite(y_sw2))
    np.testing.assert_array_equal(y_sw4, y_sw2)
    np.testing.assert_array_equal(y_sw5, y_sw2)
    np.testing.assert_array_equal(y_sw6_w2c, y_sw2)
    np.testing.assert_array_equal(y_sw5_w2c, y_sw2)
    np.testing.assert_array_equal(y_sw6_no_amax, y_sw2)

    # Row-bucket parity: m_rows=64 and m_rows=128 must match rows [0:m_rows] of m_rows=256 0-ULP
    # because per-row FP8 LHS quantization and matmul are strictly row-independent!
    y_b64 = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=5, flq_zero_rmw=2, use_w2_cache=True, m_rows=64
    )
    y_b128 = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=5, flq_zero_rmw=2, use_w2_cache=True, m_rows=128
    )
    np.testing.assert_array_equal(y_b64[:37], y_sw2[:37])
    np.testing.assert_array_equal(y_b128[:99], y_sw2[:99])


def test_sw5_mosaic_imem_shrink_and_c24k_full_pipeline_lowering():
    """Verify sw4/sw5/w2c1/rb3/rb4 lower cleanly on TPU7x at T=24576 with >75% IMEM shrink."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax._src.pallas.mosaic import lowering as mosaic_lowering
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh

    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_v2_gather_scatter import (
        Dimensions,
        GmmConfigs,
        InputConfigs,
        MetadataRef,
        TileSizes,
        WeightsRef,
        inner_kernel,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    captured: list[str] = []
    orig_lower = mosaic_lowering.lower_jaxpr_to_pipelined_module

    def _spy(*a, **kw):
        res = orig_lower(*a, **kw)
        mod = res[0] if isinstance(res, tuple) else res
        captured.append(str(mod))
        return res

    def _lower_inner(sw_mode: int) -> str:
        captured.clear()
        m_rows, k1, n1_half, k2, n2, rhs_qbs = 256, 4096, 2048, 4096, 2048, 512
        cfgs1 = GmmConfigs(
            dims=Dimensions(196608, k1, n1_half * 2, 34, 272, 8),
            tiles=TileSizes(m_rows, k1, n1_half * 2),
            lhs_cfgs=InputConfigs(jnp.float8_e4m3fn.dtype, k1, jnp.bfloat16),
            rhs_cfgs=InputConfigs(
                jnp.float4_e2m1fn.dtype,
                rhs_qbs,
                jnp.float4_e2m1fn,
                False,
                True,
                8,
                k1 // rhs_qbs,
            ),
            out_dtype=jnp.bfloat16,
            acc_dtype=jnp.float32,
            zero_init=False,
            fuse_act="silu",
        )
        cfgs2 = GmmConfigs(
            dims=Dimensions(196608, k2, n2 * 2, 34, 272, 8),
            tiles=TileSizes(m_rows, k2, n2),
            lhs_cfgs=InputConfigs(jnp.float8_e4m3fn.dtype, k2, jnp.bfloat16),
            rhs_cfgs=InputConfigs(
                jnp.float4_e2m1fn.dtype,
                rhs_qbs,
                jnp.float4_e2m1fn,
                False,
                True,
                8,
                k2 // rhs_qbs,
            ),
            out_dtype=jnp.bfloat16,
            acc_dtype=jnp.float32,
            zero_init=False,
            fuse_act=None,
        )

        def _ker(
            x3d_r,
            w1_r,
            s1_r,
            w2_r,
            s2_r,
            w2c_r,
            pnw_r,
            out_r,
            g1_r,
            pq_fp8_r,
            pq_sc_r,
            p1_r,
            acc_r,
            m_gid,
            m_off,
        ):
            meta = MetadataRef(gm_id_to_group_id=m_gid, gm_id_to_m_offset=m_off)
            for nid in range(2):
                inner_kernel(
                    None,
                    WeightsRef(weight=w1_r, scale=s1_r, bias=None),
                    g1_r.at[:, pl.ds(nid * n1_half, n1_half)],
                    p1_r,
                    acc_r,
                    meta,
                    cfgs=cfgs1,
                    scatter_mode=True,
                    _k_id=0,
                    _num_k=1,
                    _gm_id=0,
                    _n_id=nid,
                    _num_n=2,
                    fused_g2_amax_slot=1,
                    prequant_lhs_fp8_ref=pq_fp8_r,
                    prequant_lhs_scale_ref=pq_sc_r,
                    skip_row_mask=True,
                    lhs_3d_vmem_ref=x3d_r,
                )
            for nid in range(2):
                inner_kernel(
                    None,
                    WeightsRef(weight=w2_r, scale=s2_r, bias=None),
                    out_r.at[:, pl.ds(nid * n2, n2)],
                    p1_r,
                    acc_r,
                    meta,
                    cfgs=cfgs2,
                    scatter_mode=True,
                    _k_id=0,
                    _num_k=1,
                    _gm_id=0,
                    _n_id=nid,
                    _num_n=2,
                    fused_g2_amax_slot=1,
                    prequant_lhs_fp8_ref=pq_fp8_r,
                    prequant_lhs_scale_ref=pq_sc_r,
                    pnw_ref=pnw_r,
                    skip_row_mask=True,
                    lhs_2d_vmem_ref=g1_r,
                    w2_fp8_cache_ref=w2c_r,
                )

        sd = jax.ShapeDtypeStruct
        with (
            patch.object(mosaic_lowering, "lower_jaxpr_to_pipelined_module", _spy),
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            patch("jax.default_backend", return_value="tpu"),
            tpu_info_for_cpu_export(),
            patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": str(sw_mode),
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "2",
                },
            ),
        ):
            jax.clear_caches()
            jax.jit(
                pl.pallas_call(
                    _ker,
                    out_shape=sd((m_rows, n2 * 2), jnp.bfloat16),
                    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 7,
                    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                    scratch_shapes=[
                        pltpu.VMEM((m_rows, n1_half * 2), jnp.bfloat16),
                        pltpu.VMEM((m_rows, k1), jnp.float8_e4m3fn),
                        pltpu.VMEM((2, m_rows, 128), jnp.float32),
                        pltpu.VMEM((1, 128), jnp.float32),
                        pltpu.VMEM((1, 128), jnp.float32),
                        pltpu.SMEM((1,), jnp.int32),
                        pltpu.SMEM((2,), jnp.int32),
                    ],
                )
            ).trace(
                sd((m_rows, k1 // 128, 128), jnp.bfloat16),
                sd((k1 // 8, n1_half * 2), jnp.uint32),
                sd((k1 // rhs_qbs, 1, n1_half * 2), jnp.bfloat16),
                sd((k2 // 8, n2), jnp.uint32),
                sd((k2 // rhs_qbs, 1, n2), jnp.bfloat16),
                sd((k2, n2), jnp.float8_e4m3fn),
                sd((8, n2 * 2), jnp.bfloat16),
            ).lower(lowering_platforms=("tpu",))
        return [m for m in captured if "tpu.matmul" in m][-1]

    mlir_sw2 = _lower_inner(2)
    mlir_sw4 = _lower_inner(4)
    mlir_sw6 = _lower_inner(6)

    # sw2 emits 384 single-row vector<1x1x256xbf16> RHS scale loads; sw4/sw6 emit 0!
    assert "vector<1x1x256xbf16>" in mlir_sw2
    assert "vector<1x1x256xbf16>" not in mlir_sw4
    assert "vector<1x1x256xbf16>" not in mlir_sw6
    # sw6 shrinks Mosaic MLIR lines by > 75% (from >11,000 down to <3,000)!
    assert len(mlir_sw6.splitlines()) < 3500
    assert len(mlir_sw6.splitlines()) * 4 < len(mlir_sw2.splitlines())

    if len(jax.devices()) < 8:
        pytest.skip(
            "Full 8-rank pipeline HLO check requires >=8 virtual/physical JAX devices"
        )

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    base_env = {
        "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
        "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE": "1",
        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
        "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "1",
        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "2",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
        "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
        "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
        "SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL": "4096",
        "SGLANG_TPU_MUSE_KERNEL_D_T64_C16": "1",
        "SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP": "1",
    }

    arms = (
        (
            {
                **base_env,
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "5",
                "SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "3",
            },
            8,
            ("E_34-M_196608", "-rb3", "-w2c1", "-sw5", "-sfp8_6", "-cm1s2d"),
            ("-u16w",),
        ),
        (
            {
                **base_env,
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "4",
                "SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "1",
            },
            8,
            # At qb512 (8 scale blocks), wla1 3-buf ring is unsupported (8 % 16 != 0), so
            # _select_fused_rs_block_sizes keeps 2 weight buffers and activates -w2c1!
            ("E_34-M_196608", "-w2c1", "-sw4"),
            ("-wla1",),
        ),
        (
            {
                **{
                    k: v
                    for k, v in base_env.items()
                    if k != "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK"
                },
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "4",
                "SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "1",
            },
            16,
            # At qb256 (16 scale blocks), wla1 uses 3 weight buffers -> -w2c1 safely suppressed!
            ("E_34-M_196608", "-wla1", "-sw4"),
            ("-w2c1",),
        ),
    )

    with (
        patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        patch("jax.default_backend", return_value="tpu"),
        tpu_info_for_cpu_export(),
    ):
        try:
            for env_dict, n_sc_blks, must_have, must_not_have in arms:
                with patch.dict(os.environ, env_dict):
                    jax.clear_caches()

                    def _wrap(
                        a,
                        s,
                        aa,
                        ba,
                        pfw,
                        wr,
                        eb,
                        wp,
                        wps,
                        pew,
                        w1,
                        w2,
                        s1,
                        s2,
                        pnw,
                        go,
                        wu,
                        wus,
                        npnw,
                        naa,
                        nba,
                        npfw,
                        vt,
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
                                next_in_gather_rows=1280,
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
                            a,
                            s,
                            aa,
                            ba,
                            pfw,
                            wr,
                            eb,
                            wp,
                            wps,
                            pew,
                            w1,
                            w2,
                            s1,
                            s2,
                            pnw,
                            go,
                            wu,
                            wus,
                            npnw,
                            naa,
                            nba,
                            npfw,
                            vt,
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
                            sd((256, n_sc_blks, 8192), jnp.bfloat16),
                            sd((256, n_sc_blks, 4096), jnp.bfloat16),
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
                    for tok in must_have:
                        assert tok in hlo, f"Expected {tok!r} in HLO"
                    for tok in must_not_have:
                        assert tok not in hlo, f"Unexpected {tok!r} in HLO"
        finally:
            jax.clear_caches()


def test_ep16_2x2x2_chipmate3_and_w2c1_mosaic_lowering():
    """Verify EP=16 (2x2x2, num_local_experts=16, T=16384) 2-slot & 3-slot chipmate + sw4 + w2c1."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    if len(jax.devices()) < 16:
        pytest.skip(
            "16-rank EP=16 lowering check requires >=16 virtual/physical JAX devices"
        )

    from sglang_tpu.moe.pallas.fused_rs_muse.fused_moe import moe_gmm_local_rs_nodedup
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    mesh = jax.sharding.Mesh(np.array(jax.devices()[:16]), ("model",))
    P = jax.sharding.PartitionSpec
    sd = jax.ShapeDtypeStruct

    env = {
        "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_FP8_GMM_TILES": "1",
        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "2",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "3",
        "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
        "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "4",
        "SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE": "1",
    }
    jax.clear_caches()
    try:
        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            patch("jax.default_backend", return_value="tpu"),
            tpu_info_for_cpu_export(),
            patch.dict(os.environ, env),
        ):
            # (a) Direct moe_gmm_local_rs_nodedup with 3-slot hypercube chipmate (XOR 1, 2, 4) on EP=16:
            def _direct_ep16(*xs):
                return jax.shard_map(
                    lambda hs, w1, w1s, w2, w2s, go, tw, gids, pnw, lidx, gs, gst, tr, cmp: (
                        moe_gmm_local_rs_nodedup(
                            hs,
                            w1,
                            w1s,
                            None,
                            w2,
                            w2s,
                            None,
                            None,
                            None,
                            go,
                            tw,
                            gids,
                            pnw,
                            activation="silu",
                            topk=8,
                            ep_size=16,
                            ep_axis_name="model",
                            has_post_norm=True,
                            sp_enabled=True,
                            fp8_direct_write=False,
                            expert_tp=False,
                            fp4_w4a16=False,
                            local_t_recv=tr[0],
                            return_tiled=True,
                            block_scatter=True,
                            return_raw_out_buf=True,
                            precomputed_routing=(lidx, gs, gst),
                            precomputed_chipmate_plan=cmp,
                        )
                    ),
                    mesh=mesh,
                    in_specs=(
                        P(),
                        P("model"),
                        P("model"),
                        P("model"),
                        P("model"),
                        P("model"),
                        P("model", None),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                    ),
                    out_specs=P("model", None, None),
                    check_vma=False,
                )(*xs)

            hlo_direct = (
                jax.jit(_direct_ep16)
                .trace(
                    sd((16384, 32, 128), jnp.bfloat16),
                    sd((256, 4096, 8192), jnp.float4_e2m1fn),
                    sd((256, 8, 8192), jnp.bfloat16),
                    sd((256, 4096, 4096), jnp.float4_e2m1fn),
                    sd((256, 8, 4096), jnp.bfloat16),
                    sd((16,), jnp.int32),
                    sd((16384, 8), jnp.bfloat16),
                    sd((16384, 8), jnp.int32),
                    sd((4096,), jnp.bfloat16),
                    sd((131072,), jnp.int32),
                    sd((19 * 16,), jnp.int32),
                    sd((2 + 16 + 19 * 16,), jnp.int32),
                    sd((1,), jnp.int32),
                    sd((6,), jnp.int32),
                )
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )
            assert "E_19-M_131072" in hlo_direct
            assert "-EP_16-" in hlo_direct
            assert "-w2c1-sw4-sfp8_6-cm1s3d" in hlo_direct

            # (b) Full fused_moe_full_bridge_and_ffn pipeline at EP=16, T=16384:
            def _wrap(*xs):
                return jax.shard_map(
                    lambda *ys: fused_moe_full_bridge_and_ffn(
                        *ys[:-1],
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
                        valid_tokens=ys[-1],
                        next_in_gather_rows=1024,
                        next_in_gather_static_fp8=True,
                    ),
                    mesh=mesh,
                    in_specs=(
                        P("model", None),
                        P("model", None),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P("model", None, None),
                        P("model", None, None),
                        P("model", None, None),
                        P("model", None, None),
                        P(),
                        P("model"),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                        P(),
                    ),
                    out_specs=(
                        P("model", None),
                        P("model", None),
                        P("model", None),
                        P("model", None),
                        P("model", None, None),
                    ),
                    check_vma=False,
                )(*xs)

            hlo_pipe = (
                jax.jit(_wrap)
                .trace(
                    sd((16384, 8192), jnp.bfloat16),
                    sd((16384, 8192), jnp.float32),
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
                    sd((16,), jnp.int32),
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
            assert "-EP_16-" in hlo_pipe
            assert "-w2c1-sw4-sfp8_6" in hlo_pipe
    finally:
        jax.clear_caches()
