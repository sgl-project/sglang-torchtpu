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


def test_eo3_eo4_w1_fp8_expert_cache_parity_and_mosaic_lowering():
    """Verify GMM_EXPERT_OUTER_FP8=4 0-ULP interpret parity across [1,3,5+2] tiles and EP=8/16 lowering."""
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
    from sglang_tpu.moe.pallas.fused_rs_muse.w1_fp8_expert_cache import (
        compute_eo3_batch_count,
        run_eo3_expert_batch,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    rng = np.random.default_rng(20261122)
    # 11 tiles: E0=1 tile (0), E1=3 tiles (1..3), E2=7 tiles (4..10 -> split 5 + 2!)
    num_gm = 11
    e_ids = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0], dtype=np.int32)
    m_offs = np.arange(num_gm + 1, dtype=np.int32) * 256
    num_e = 3
    tile_m, k1, n1_half, k2, n2, rhs_qbs = 256, 2048, 512, 1024, 512, 256

    x_all = jnp.asarray(
        rng.standard_normal((num_gm, tile_m, k1 // 128, 128)).astype(np.float32) * 0.5,
        jnp.bfloat16,
    )
    w1_u32 = jnp.asarray(
        rng.integers(
            0, 2**31 - 1, size=(num_e, 2, k1 // 8, n1_half * 2), dtype=np.uint32
        )
    )
    s1_bf16 = jnp.asarray(
        np.exp(
            rng.normal(-2.0, 0.25, size=(num_e, 2, k1 // rhs_qbs, 1, n1_half * 2))
        ).astype(np.float32),
        jnp.bfloat16,
    )
    w2_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(num_e, 2, k2 // 8, n2), dtype=np.uint32)
    )
    s2_bf16 = jnp.asarray(
        np.exp(rng.normal(-2.0, 0.25, size=(num_e, 2, k2 // rhs_qbs, 1, n2))).astype(
            np.float32
        ),
        jnp.bfloat16,
    )
    pnw_1d = jnp.asarray(
        1.0 + 0.1 * rng.standard_normal((n2 * 2,)).astype(np.float32), jnp.bfloat16
    )
    pnw_8 = jnp.broadcast_to(pnw_1d[None, :], (8, n2 * 2))
    gs_g = jnp.asarray([0.85, 1.05, 0.95], dtype=jnp.float32)
    gs_u = jnp.asarray([1.15, 0.90, 1.10], dtype=jnp.float32)
    gs_w2 = jnp.asarray([0.92, 1.08, 0.97], dtype=jnp.float32)

    def _run_ref_sw4(raw_bf16: bool = True) -> np.ndarray:
        cfgs1 = GmmConfigs(
            dims=Dimensions(196608, k1, n1_half * 2, num_e, num_e, 8),
            tiles=TileSizes(tile_m, k1, n1_half),
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
            dims=Dimensions(196608, k2, n2 * 2, num_e, num_e, 8),
            tiles=TileSizes(tile_m, k2, n2),
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
            x_r,
            w1_r,
            s1_r,
            w2_r,
            s2_r,
            pnw_r,
            eid_r,
            moff_r,
            gsg_r,
            gsu_r,
            gsw2_r,
            out_r,
            g1_r,
            pq_r,
            pqsc_r,
            p1_r,
            acc_r,
            fg_r,
            fm_r,
        ):

            meta = MetadataRef(gm_id_to_group_id=fg_r, gm_id_to_m_offset=fm_r)
            fm_r[0] = jnp.int32(0)
            fm_r[1] = jnp.int32(tile_m)
            for gm in range(num_gm):
                e = eid_r[gm]
                fg_r[0] = e
                for nid in range(2):
                    inner_kernel(
                        None,
                        WeightsRef(
                            weight=w1_r.at[e, nid], scale=s1_r.at[e, nid], bias=None
                        ),
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
                        prequant_lhs_fp8_ref=pq_r,
                        prequant_lhs_scale_ref=pqsc_r,
                        skip_row_mask=True,
                        lhs_3d_vmem_ref=x_r.at[gm],
                    )
                for nid in range(2):
                    inner_kernel(
                        None,
                        WeightsRef(
                            weight=w2_r.at[e, nid], scale=s2_r.at[e, nid], bias=None
                        ),
                        out_r.at[gm, :, pl.ds(nid * n2, n2)],
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
                        prequant_lhs_fp8_ref=pq_r.at[:, :k2],
                        prequant_lhs_scale_ref=pqsc_r,
                        pnw_ref=pnw_r,
                        skip_row_mask=True,
                        lhs_2d_vmem_ref=g1_r,
                    )
                if not raw_bf16:
                    y_bf16 = out_r[gm][...]
                    var = jnp.sum(
                        y_bf16.astype(jnp.float32) ** 2, axis=1, keepdims=True
                    ) * (1.0 / (n2 * 2))
                    inv_rms = (jax.lax.rsqrt(var + 1e-8) * jnp.float32(6.0)).astype(
                        jnp.bfloat16
                    )
                    out_r[gm] = (
                        (y_bf16 * inv_rms)
                        .astype(jnp.float8_e4m3fn)
                        .astype(jnp.bfloat16)
                    )

        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
            patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "4",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": str(k1),
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                },
            ),
        ):
            return np.asarray(
                pl.pallas_call(
                    _ker,
                    out_shape=jax.ShapeDtypeStruct(
                        (num_gm, tile_m, n2 * 2), jnp.bfloat16
                    ),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 6
                        + [pl.BlockSpec(memory_space=pltpu.SMEM)] * 5,
                        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                        scratch_shapes=[
                            pltpu.VMEM((tile_m, n1_half * 2), jnp.bfloat16),
                            pltpu.VMEM((tile_m, k1), jnp.float8_e4m3fn),
                            pltpu.VMEM((1, tile_m, 128), jnp.float32),
                            pltpu.VMEM((1, 128), jnp.float32),
                            pltpu.VMEM((1, 128), jnp.float32),
                            pltpu.SMEM((1,), jnp.int32),
                            pltpu.SMEM((2,), jnp.int32),
                        ],
                    ),
                    interpret=True,
                )(
                    x_all,
                    w1_u32,
                    s1_bf16,
                    w2_u32,
                    s2_bf16,
                    pnw_8,
                    jnp.asarray(e_ids),
                    jnp.asarray(m_offs),
                    gs_g,
                    gs_u,
                    gs_w2,
                ),
                dtype=np.float32,
            )

    def _run_eo_batch(direct_fp8: bool) -> np.ndarray:
        def _ker(
            x_r,
            w1_r,
            s1_r,
            w2_r,
            s2_r,
            pnw_r,
            eid_r,
            moff_r,
            gsg_r,
            gsu_r,
            gsw2_r,
            out_r,
            g_lhs_2x,
            w1_b,
            s1_b,
            w2_b,
            s2_b,
            stg_4x,
            wh_r,
            wc_r,
            lfp8_5x,
            lsc_5x,
            buf_5x,
            gmid_r,
            fg_r,
            fm_r,
            gsem_r,
        ):
            meta = MetadataRef(gm_id_to_group_id=eid_r, gm_id_to_m_offset=moff_r)
            fmeta = MetadataRef(gm_id_to_group_id=fg_r, gm_id_to_m_offset=fm_r)

            def _g_start(hs, dst, idx, sem, gm, mref, divisor=1, packed_u16=False):
                dst[...] = x_r[gm]

            def _noop(*_a, **_kw):
                pass

            def _w1_start(slot, eid, nid, kid):
                w1_b[slot] = w1_r[eid, nid]
                s1_b[slot] = s1_r[eid, nid]

            def _w2_start(slot, eid, nid, kid):
                w2_b[slot] = w2_r[eid, nid]
                s2_b[slot] = s2_r[eid, nid]

            def _direct_w(gm, slot):
                out_r[gm] = stg_4x[slot].reshape(tile_m, n2 * 2).astype(jnp.bfloat16)

            def _body(gm_id, skip_cur):
                should_run = skip_cur == 0
                bn = compute_eo3_batch_count(
                    gm_id, jnp.int32(num_gm), meta, max_batch=5
                )
                nxt_skip = jnp.where(should_run, bn - 1, jnp.maximum(skip_cur - 1, 0))

                @pl.when(should_run)
                def _():
                    run_eo3_expert_batch(
                        gm_id=gm_id,
                        batch_n=bn,
                        local_num_gm=jnp.int32(num_gm),
                        gm_id_ref=gmid_r,
                        metadata_ref=meta,
                        fused_metadata_ref=fmeta,
                        hidden_states_ref=x_r,
                        gathered_lhs_2x_ref=g_lhs_2x,
                        lhs_indices_ref=eid_r,
                        gather_sem_ref=gsem_r,
                        gather_divisor=1,
                        packed_u16_lhs=False,
                        dma_gather_gm_start_fn=_g_start,
                        dma_gather_gm_wait_fn=_noop,
                        start_w1_dma_fn=_w1_start,
                        wait_w1_dma_fn=_noop,
                        start_w2_dma_fn=_w2_start,
                        wait_w2_dma_fn=_noop,
                        wait_staging_slot_writes_fn=_noop,
                        do_direct_write_fn=_direct_w,
                        maybe_start_deferred_cm_wt_send_fn=None,
                        enable_chipmate=False,
                        cm_defer_send=False,
                        defer_writes=True,
                        w1_buf_ref=w1_b,
                        w1_scale_buf_ref=s1_b,
                        w2_buf_ref=w2_b,
                        w2_scale_buf_ref=s2_b,
                        w1_gs_gate_ref=gsg_r,
                        w1_gs_up_ref=gsu_r,
                        w2_gs_ref=gsw2_r,
                        has_w1_gs=True,
                        has_w2_gs=True,
                        pnw_ref=pnw_r,
                        in_kernel_post_norm_size=n2 * 2,
                        static_fp8_rs_scale=6.0,
                        scatter_staging_3x_ref=stg_4x,
                        eo_w_half_fp8_ref=wh_r,
                        eo_w_colc_ref=wc_r,
                        eo_lhs_fp8_4x_ref=lfp8_5x,
                        eo_lhs_sc_4x_ref=lsc_5x,
                        eo_buf_4x_ref=buf_5x,
                        num_n1=2,
                        num_n2=2,
                        tile_m=tile_m,
                        tile_k1=k1,
                        tile_n1=n1_half,
                        tile_k2=k2,
                        tile_n2=n2,
                        strip_step=256,
                        rhs1_qbs=rhs_qbs,
                        rhs2_qbs=rhs_qbs,
                        packing1=8,
                        packing2=8,
                        rhs1_quant_dtype=jnp.float4_e2m1fn.dtype,
                        rhs2_quant_dtype=jnp.float4_e2m1fn.dtype,
                        out_dtype=jnp.bfloat16,
                        direct_fp8_mode=direct_fp8,
                        use_bf16_inner=True,
                        k_chunk=256,
                    )

                return nxt_skip

            jax.lax.fori_loop(0, num_gm, _body, jnp.int32(0))

        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            return np.asarray(
                pl.pallas_call(
                    _ker,
                    out_shape=jax.ShapeDtypeStruct(
                        (num_gm, tile_m, n2 * 2), jnp.bfloat16
                    ),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 6
                        + [pl.BlockSpec(memory_space=pltpu.SMEM)] * 5,
                        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                        scratch_shapes=[
                            pltpu.VMEM((2, tile_m, k1 // 128, 128), jnp.bfloat16),
                            pltpu.VMEM((2, k1 // 8, n1_half * 2), jnp.uint32),
                            pltpu.VMEM(
                                (2, k1 // rhs_qbs, 1, n1_half * 2), jnp.bfloat16
                            ),
                            pltpu.VMEM((2, k2 // 8, n2), jnp.uint32),
                            pltpu.VMEM((2, k2 // rhs_qbs, 1, n2), jnp.bfloat16),
                            pltpu.VMEM(
                                (4, tile_m, (n2 * 2) // 128, 128), jnp.float8_e4m3fn
                            ),
                            pltpu.VMEM((k1, n1_half), jnp.float8_e4m3fn),
                            pltpu.VMEM((8, n1_half), jnp.float32),
                            pltpu.VMEM((5, tile_m, k1), jnp.float8_e4m3fn),
                            pltpu.VMEM((5, tile_m, 128), jnp.float32),
                            pltpu.VMEM((5, tile_m, n1_half * 2), jnp.bfloat16),
                            pltpu.SMEM((1,), jnp.int32),
                            pltpu.SMEM((1,), jnp.int32),
                            pltpu.SMEM((2,), jnp.int32),
                            pltpu.SMEM((2,), jnp.int32),
                        ],
                    ),
                    interpret=True,
                )(
                    x_all,
                    w1_u32,
                    s1_bf16,
                    w2_u32,
                    s2_bf16,
                    pnw_8,
                    jnp.asarray(e_ids),
                    jnp.asarray(m_offs),
                    gs_g,
                    gs_u,
                    gs_w2,
                ),
                dtype=np.float32,
            )

    y_ref = _run_ref_sw4(raw_bf16=False)
    y_eo4 = _run_eo_batch(direct_fp8=True)
    y_eo3 = _run_eo_batch(direct_fp8=False)
    assert np.all(np.isfinite(y_ref))
    # 0.0-ULP bit-exact parity across all 11 tiles ([1, 3, 5+2])!
    np.testing.assert_array_equal(y_eo4, y_ref)
    cos_eo3 = float(
        np.sum(y_eo3 * y_ref) / (np.linalg.norm(y_eo3) * np.linalg.norm(y_ref))
    )
    assert cos_eo3 > 0.9975, f"cos_eo3={cos_eo3}"

    if len(jax.devices()) < 8:
        return
    mesh8 = jax.sharding.Mesh(np.array(jax.devices()[:8]), ("model",))
    P = jax.sharding.PartitionSpec
    sd = jax.ShapeDtypeStruct
    for eo_val, expected_tag in (("3", "-eo3-sfp8_6"), ("4", "-eo4-sfp8_6")):
        env = {
            "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
            "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
            "SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE": "1",
            "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
            "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
            "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "1",
            "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_FP8_GMM_TILES": "1",
            "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
            "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
            "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
            "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "2",
            "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "4",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
            "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
            "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
            "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "4",
            "SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8": eo_val,
        }
        jax.clear_caches()
        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            patch("jax.default_backend", return_value="tpu"),
            tpu_info_for_cpu_export(),
            patch.dict(os.environ, env),
        ):
            hlo = (
                jax.jit(
                    lambda *xs: jax.shard_map(
                        lambda *ys: fused_moe_full_bridge_and_ffn(
                            *ys[:-1],
                            mesh=mesh8,
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
                            next_in_gather_rows=1280,
                            next_in_gather_static_fp8=True,
                        ),
                        mesh=mesh8,
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
                )
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
            assert expected_tag in hlo
    jax.clear_caches()


@pytest.mark.parametrize(
    ("m_rows", "k_dim", "rhs_qbs"),
    [
        (256, 2048, 256),
        (128, 2048, 256),
        (256, 2048, 512),
    ],
)
def test_sw7_pair_k512_and_fori_tail_bit_exact_vs_sw2(
    m_rows: int, k_dim: int, rhs_qbs: int
) -> None:
    """GMM_SCOPED_W4A8=7 (mh_outer=False + hoisted scales + 512-K pair unpack + m<256 fori)
    must be 0.0-ULP bit-exact with sw=2 across full (m=256) and tail (m=128) buckets."""
    ref_sw2 = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=2,
        flq_zero_rmw=3,
        use_w2_cache=False,
        m_rows=m_rows,
        k1=k_dim,
        n1_half=k_dim // 2,
        k2=k_dim,
        n2=512,
        rhs_qbs=rhs_qbs,
        seed=20261123 + m_rows + rhs_qbs,
        input_rows=max(256, m_rows),
    )
    got_sw7 = _run_two_ntile_gmm1_gmm2_interpret(
        sw_mode=7,
        flq_zero_rmw=3,
        use_w2_cache=False,
        m_rows=m_rows,
        k1=k_dim,
        n1_half=k_dim // 2,
        k2=k_dim,
        n2=512,
        rhs_qbs=rhs_qbs,
        seed=20261123 + m_rows + rhs_qbs,
        input_rows=max(256, m_rows),
    )
    assert np.array_equal(got_sw7, ref_sw2), (
        f"sw=7 mismatch vs sw=2 at m_rows={m_rows}, k_dim={k_dim}, rhs_qbs={rhs_qbs}: "
        f"max_abs={np.max(np.abs(got_sw7.astype(np.float32) - ref_sw2.astype(np.float32)))}"
    )
