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
            interpret=pltpu.InterpretParams(uninitialized_memory="nan"),
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
                    interpret=pltpu.InterpretParams(uninitialized_memory="nan"),
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
                    interpret=pltpu.InterpretParams(uninitialized_memory="nan"),
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


def test_ep16_auto_smem_window_and_xor_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    """EP>=16 at M>=131072 must auto-window SMEM to 65536 rows so secondary_idx_ref (50 ints/tile)
    never exhausts SMEM and triggers scalar-register-to-HBM spills, while EP=8 stays at 131072."""
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        fused_rs_smem_rows,
    )

    monkeypatch.setenv("SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS", "0")
    monkeypatch.setenv("SGLANG_TPU_MUSE_KERNEL_GMM_SMEM_WINDOW_ROWS", "0")
    assert fused_rs_smem_rows(131_072, 8, False) == 131_072
    assert fused_rs_smem_rows(131_072, 16, False) == 65_536
    assert fused_rs_smem_rows(196_608, 16, False) == 65_536
    assert fused_rs_smem_rows(262_144, 16, False) == 131_072

    for ep_size, expected in (
        (8, (2, 4, 6, 3, 5, 7, 1)),
        (16, (2, 4, 6, 8, 10, 12, 14, 3, 5, 7, 9, 11, 13, 15, 1)),
    ):
        got = (
            tuple(range(2, ep_size, 2))
            + tuple(range(3, ep_size, 2))
            + ((1,) if ep_size >= 2 else ())
        )
        assert got == expected
        assert len(set(got)) == ep_size - 1
        assert got[-1] == 1


@pytest.mark.parametrize("sortfree_prefill", ["1", "0"])
def test_compute_block_scatter_routing_rotate_rs2_with_pallas_sf0_bit_exact(
    monkeypatch: pytest.MonkeyPatch, sortfree_prefill: str
) -> None:
    """When PALLAS_SORTFREE_ROUTING=0 (precomputed_counts=None), GMM_ROTATE_RS=2 must still
    rotate lhs_indices consistently with _precompute_block_scatter_meta_closed_form(rotate_rs=True)."""
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        _precompute_block_scatter_meta_closed_form,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        compute_block_scatter_routing,
    )

    monkeypatch.setenv("SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING", "0")
    monkeypatch.setenv("SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS", "2")
    monkeypatch.setenv("SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL", sortfree_prefill)

    ep_size = 16
    num_local_experts = 4
    num_experts = ep_size * num_local_experts
    topk = 8
    num_tokens = 1024
    chunk_size = num_tokens // ep_size
    tile_m = 64

    rng = np.random.default_rng(20261123)
    global_topk = np.stack(
        [rng.choice(num_experts, size=topk, replace=False) for _ in range(num_tokens)],
        axis=0,
    ).astype(np.int32)
    global_topk_j = jnp.asarray(global_topk)

    for my_id_int in (0, 1, 7, 15):
        go_val = jnp.int32(my_id_int * num_local_experts)
        my_id = jnp.int32(my_id_int)
        lhs_idx, group_sizes, group_start_arg, _ = compute_block_scatter_routing(
            global_topk_j,
            go_val,
            my_id,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            topk=topk,
            precomputed_counts=None,
        )
        num_slots = num_local_experts + (num_tokens * topk) // tile_m
        meta = np.asarray(
            _precompute_block_scatter_meta_closed_form(
                group_sizes,
                jnp.asarray([go_val], dtype=jnp.int32),
                group_start_arg,
                size_group=num_local_experts,
                tile_m=tile_m,
                num_slots=num_slots,
                ep_size=ep_size,
                rotate_rs=True,
            )
        )
        lhs_np = np.asarray(lhs_idx)
        local_start = int(np.asarray(group_start_arg)[0])
        gs_local = np.asarray(group_sizes)[
            my_id_int * num_local_experts : (my_id_int + 1) * num_local_experts
        ]
        cur_m = local_start
        gm_slot = 0
        for g_sz in gs_local:
            rem = int(g_sz)
            while rem > 0:
                take = min(tile_m, rem)
                tile_tokens = lhs_np[cur_m : cur_m + take]
                covered = 0
                for c in range(ep_size):
                    s_c = int(meta[gm_slot, c])
                    l_c = int(meta[gm_slot, 2 * ep_size + c])
                    if l_c > 0:
                        seg = tile_tokens[s_c : s_c + l_c]
                        assert np.all(seg // chunk_size == c), (
                            f"rank={my_id_int} gm={gm_slot} c={c}: expected chip {c}, "
                            f"got chips {seg // chunk_size}"
                        )
                        covered += l_c
                assert covered == take
                cur_m += take
                rem -= take
                gm_slot += 1


@pytest.mark.parametrize("m_rows", [256, 128])
def test_w9_pql1_pql2_sw8_nan_poison_interpret_zero_ulp(m_rows: int) -> None:
    """Verify 1x-per-tile FP8 unpack (pql1/pql2), 3-level binary-tree scale fill,
    and fused_g2_amax_slot=1 (with NaN-poisoned slot 1) match pq0+sw2 at 0.0-ULP."""
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
    from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
        fill_pq_lhs_scales_tree,
        unpack_3d_pq_lhs_to_2d_fp8,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    k1, n1_half, k2, n2, rhs_qbs = 2048, 512, 1024, 512, 512
    rng = np.random.default_rng(20261124 + m_rows)
    lhs_3d_bf16 = jnp.asarray(
        rng.standard_normal((m_rows, k1 // 128, 128)).astype(np.float32) * 0.5,
        jnp.bfloat16,
    )
    x_bf16_2d = lhs_3d_bf16.reshape(m_rows, k1)
    amax = jnp.max(jnp.abs(x_bf16_2d), axis=1, keepdims=True).astype(jnp.float32)
    inv_m = jnp.where(amax == 0, 0.0, jnp.float32(448.0) / amax).astype(jnp.bfloat16)
    row_sc = jnp.where(amax == 0, 0.0, amax * jnp.float32(1.0 / 448.0)).reshape(-1)
    lhs_3d_fp8 = (
        (x_bf16_2d * inv_m).astype(jnp.float8_e4m3fn).reshape(m_rows, k1 // 128, 128)
    )
    perm = rng.permutation(m_rows).astype(np.int32)
    inv_perm = np.argsort(perm).astype(np.int32)
    hs_sc_perm = jnp.asarray(np.asarray(row_sc)[inv_perm], jnp.float32)
    lhs_idx_perm = jnp.asarray(perm, jnp.int32)
    u16_words = (perm[0::2].astype(np.uint32) & 0xFFFF) | (
        (perm[1::2].astype(np.uint32) & 0xFFFF) << 16
    )
    lhs_idx_u16 = jnp.asarray(u16_words.view(np.int32), jnp.int32)

    w1_a = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    w1_b = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    s1_a = jnp.asarray(
        np.exp(rng.normal(-2.5, 0.2, size=(k1 // rhs_qbs, 1, n1_half * 2))).astype(
            np.float32
        ),
        jnp.bfloat16,
    )
    s1_b = jnp.asarray(
        np.exp(rng.normal(-2.5, 0.2, size=(k1 // rhs_qbs, 1, n1_half * 2))).astype(
            np.float32
        ),
        jnp.bfloat16,
    )
    w2_a = jnp.asarray(rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32))
    w2_b = jnp.asarray(rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32))
    s2_a = jnp.asarray(
        np.exp(rng.normal(-2.5, 0.2, size=(k2 // rhs_qbs, 1, n2))).astype(np.float32),
        jnp.bfloat16,
    )
    s2_b = jnp.asarray(
        np.exp(rng.normal(-2.5, 0.2, size=(k2 // rhs_qbs, 1, n2))).astype(np.float32),
        jnp.bfloat16,
    )
    pnw_8 = jnp.broadcast_to(
        jnp.asarray(
            1.0 + 0.1 * rng.standard_normal((n2 * 2,)).astype(np.float32), jnp.bfloat16
        )[None, :],
        (8, n2 * 2),
    )
    gs_gate = jnp.asarray([0.85], dtype=jnp.float32)
    gs_up = jnp.asarray([1.15], dtype=jnp.float32)
    gs_w2 = jnp.asarray([0.92], dtype=jnp.float32)

    cfgs1 = GmmConfigs(
        dims=Dimensions(
            size_m=131072,
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
            size_m=131072,
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

    def _run(
        sw_mode: int,
        pq_mode: int,
        packed_u16: bool = False,
        num_valid: int | None = None,
        skip_row_mask: bool = True,
    ) -> np.ndarray:
        nv = m_rows if num_valid is None else num_valid
        idx_arg = lhs_idx_u16 if packed_u16 else lhs_idx_perm

        def _ker(
            sc_smem_r,
            idx_smem_r,
            x_bf16_r,
            x_fp8_r,
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
        ):
            m_gid[0] = jnp.int32(0)
            m_off[0] = jnp.int32(0)
            m_off[1] = jnp.int32(nv)
            meta = MetadataRef(gm_id_to_group_id=m_gid, gm_id_to_m_offset=m_off)
            if pq_mode >= 1:
                fill_pq_lhs_scales_tree(
                    pq_sc_r,
                    sc_smem_r,
                    idx_smem_r,
                    m_st=jnp.int32(0),
                    m_en=jnp.int32(nv),
                    num_valid=jnp.int32(nv),
                    num_lanes=128,
                    packed_u16_lhs=packed_u16,
                    gather_divisor=1,
                    scale_b_id=0,
                    unroll_m_rows=(m_rows if pq_mode >= 3 else None),
                )
                unpack_3d_pq_lhs_to_2d_fp8(
                    x_fp8_r,
                    pq_fp8_r,
                    tile_m=m_rows,
                    tile_k=k1,
                    via_bf16_xpose=(pq_mode == 2),
                )
            for nid, (wr, sr) in enumerate([(w1a_r, s1a_r), (w1b_r, s1b_r)]):
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
                    fused_g2_amax_slot=1,
                    prequant_lhs_fp8_ref=pq_fp8_r,
                    prequant_lhs_scale_ref=pq_sc_r,
                    skip_row_mask=skip_row_mask,
                    lhs_3d_vmem_ref=(None if pq_mode >= 1 else x_bf16_r),
                    prequant_lhs_ready=(pq_mode >= 1),
                )
            for nid, (wr, sr) in enumerate([(w2a_r, s2a_r), (w2b_r, s2b_r)]):
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
                    fused_g2_amax_slot=1,
                    prequant_lhs_fp8_ref=pq_fp8_r.at[:, :k2],
                    prequant_lhs_scale_ref=pq_sc_r,
                    pnw_ref=pnw_r,
                    skip_row_mask=skip_row_mask,
                    lhs_2d_vmem_ref=g1_r,
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
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": str(sw_mode),
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": str(k1),
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                },
            ),
        ):
            res = pl.pallas_call(
                _ker,
                out_shape=jax.ShapeDtypeStruct((m_rows, n2 * 2), jnp.bfloat16),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=2,
                    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 14,
                    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                    scratch_shapes=[
                        pltpu.VMEM((m_rows, k2), jnp.bfloat16),
                        pltpu.VMEM((m_rows, k1), jnp.float8_e4m3fn),
                        pltpu.VMEM((2, m_rows, 128), jnp.float32),
                        pltpu.VMEM((1, 128), jnp.bfloat16),
                        pltpu.VMEM((1, 128), jnp.float32),
                        pltpu.SMEM((1,), jnp.int32),
                        pltpu.SMEM((2,), jnp.int32),
                    ],
                ),
                interpret=pltpu.InterpretParams(uninitialized_memory="nan"),
            )(
                hs_sc_perm,
                idx_arg,
                lhs_3d_bf16,
                lhs_3d_fp8,
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

    ref = _run(sw_mode=2, pq_mode=0)
    assert np.isfinite(ref).all() and np.linalg.norm(ref) > 0
    for got in (
        _run(sw_mode=8, pq_mode=0),
        _run(sw_mode=2, pq_mode=1, packed_u16=False),
        _run(sw_mode=2, pq_mode=1, packed_u16=True),
        _run(sw_mode=2, pq_mode=2, packed_u16=False),
        _run(sw_mode=2, pq_mode=3, packed_u16=False),
        _run(sw_mode=2, pq_mode=3, packed_u16=True),
        _run(sw_mode=8, pq_mode=1, packed_u16=True),
    ):
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got, ref)

    # Partial-tile (num_valid < 32 < m_rows) under uninitialized_memory="nan":
    nv_part = 21
    ref_part = _run(sw_mode=2, pq_mode=0, num_valid=nv_part, skip_row_mask=False)
    assert np.isfinite(ref_part).all()
    assert np.all(ref_part[nv_part:] == 0.0)
    for pq_m in (1, 3):
        got_mask = _run(sw_mode=2, pq_mode=pq_m, num_valid=nv_part, skip_row_mask=False)
        assert np.isfinite(got_mask).all()
        np.testing.assert_array_equal(got_mask, ref_part)
        got_nomask = _run(sw_mode=2, pq_mode=pq_m, num_valid=nv_part, skip_row_mask=True)
        assert np.isfinite(got_nomask[:nv_part]).all()
        np.testing.assert_array_equal(got_nomask[:nv_part], ref[:nv_part])


@pytest.mark.parametrize("borrowed_at_gm0", [False, True])
def test_w9_g2a6_prologue_w_dma_and_split_chipmate_interpret_zero_ulp(
    borrowed_at_gm0: bool,
) -> None:
    """Verify 2-device g2a6 prologue cold-start W1/W2 DMA + split chipmate W1/W2 remote wait
    with InterpretParams(detect_races=True): 0 races, 0 leaked semaphores, 0.0-ULP vs g2a4."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices()) < 2:
        pytest.skip("Requires >=2 virtual/physical JAX devices")

    rng = np.random.default_rng(77 + int(borrowed_at_gm0))
    w1_data = jnp.asarray(
        rng.standard_normal((2, 3, 16, 128), dtype=np.float32), jnp.bfloat16
    )
    w2_data = jnp.asarray(
        rng.standard_normal((2, 3, 16, 128), dtype=np.float32), jnp.bfloat16
    )
    x_data = jnp.asarray(
        rng.standard_normal((2, 2, 16, 128), dtype=np.float32), jnp.bfloat16
    )
    mesh = Mesh(np.array(jax.devices()[:2]), ("ep",))

    def _run(split_mode: bool) -> np.ndarray:
        cm_flag_len = 5 if split_mode else 3

        def _ker(
            x_hbm,
            w1_hbm,
            w2_hbm,
            out_hbm,
            cm_w1_pad,
            cm_w2_pad,
            x_vmem,
            w1_buf,
            w2_buf,
            g_sem,
            w1_sem,
            w2_sem,
            cm_send_sem,
            cm_recv_sem,
            cm_awaited,
            bar_sem,
        ):
            my_id = jax.lax.axis_index("ep")
            partner = jax.lax.bitwise_xor(my_id, jnp.int32(1))
            borrowed_slot = jnp.int32(3)
            e0 = jnp.int32(3 if borrowed_at_gm0 else 0)
            e1 = jnp.int32(4)

            for i in range(cm_flag_len):
                cm_awaited[i] = jnp.int32(0)

            pl.semaphore_signal(
                bar_sem.at[0],
                1,
                device_id={"ep": partner},
                device_id_type=pl.DeviceIdType.MESH,
            )

            pltpu.make_async_copy(
                src_ref=x_hbm.at[0], dst_ref=x_vmem.at[0], sem=g_sem.at[0]
            ).start()
            if split_mode:

                @pl.when(e0 < borrowed_slot)
                def _pro_home():
                    pltpu.make_async_copy(
                        src_ref=w1_hbm.at[e0], dst_ref=w1_buf.at[0], sem=w1_sem.at[0]
                    ).start()
                    pltpu.make_async_copy(
                        src_ref=w2_hbm.at[e0], dst_ref=w2_buf.at[0], sem=w2_sem.at[0]
                    ).start()

            pltpu.make_async_copy(
                src_ref=x_hbm.at[1], dst_ref=x_vmem.at[1], sem=g_sem.at[1]
            ).start()
            pl.semaphore_wait(bar_sem.at[0], 1)

            def _issue_def_send():
                flag_idx = 4 if split_mode else 2

                @pl.when(cm_awaited[flag_idx] == 0)
                def _do():
                    for ci in range(2):
                        pltpu.make_async_remote_copy(
                            src_ref=w1_hbm.at[ci],
                            dst_ref=cm_w1_pad.at[ci],
                            send_sem=cm_send_sem.at[2 * ci + 0],
                            recv_sem=cm_recv_sem.at[2 * ci + 0],
                            device_id={"ep": partner},
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()
                        pltpu.make_async_remote_copy(
                            src_ref=w2_hbm.at[ci],
                            dst_ref=cm_w2_pad.at[ci],
                            send_sem=cm_send_sem.at[2 * ci + 1],
                            recv_sem=cm_recv_sem.at[2 * ci + 1],
                            device_id={"ep": partner},
                            device_id_type=pl.DeviceIdType.MESH,
                        ).start()
                    cm_awaited[flag_idx] = jnp.int32(1)

            def _wait_cm_w1(ci: int):
                @pl.when(cm_awaited[ci] == 0)
                def _w1():
                    pltpu.make_async_copy(
                        src_ref=cm_w1_pad.at[ci],
                        dst_ref=cm_w1_pad.at[ci],
                        sem=cm_recv_sem.at[2 * ci + 0],
                    ).wait()
                    if not split_mode:
                        pltpu.make_async_copy(
                            src_ref=cm_w2_pad.at[ci],
                            dst_ref=cm_w2_pad.at[ci],
                            sem=cm_recv_sem.at[2 * ci + 1],
                        ).wait()
                    cm_awaited[ci] = jnp.int32(1)

            def _wait_cm_w2(ci: int):
                if not split_mode:
                    return _wait_cm_w1(ci)

                @pl.when(cm_awaited[2 + ci] == 0)
                def _w2():
                    pltpu.make_async_copy(
                        src_ref=cm_w2_pad.at[ci],
                        dst_ref=cm_w2_pad.at[ci],
                        sem=cm_recv_sem.at[2 * ci + 1],
                    ).wait()
                    cm_awaited[2 + ci] = jnp.int32(1)

            # If gm0 is borrowed, donor must have sent first; issue send before gm0 when borrowed_at_gm0
            if borrowed_at_gm0:
                _issue_def_send()

            def _start_w(slot_b: int, eid):
                for ci in range(2):

                    @pl.when(eid == borrowed_slot + ci)
                    def _from_cm(_c=ci):
                        _wait_cm_w1(_c)
                        pltpu.make_async_copy(
                            src_ref=cm_w1_pad.at[_c],
                            dst_ref=w1_buf.at[slot_b],
                            sem=w1_sem.at[slot_b],
                        ).start()
                        _wait_cm_w2(_c)
                        pltpu.make_async_copy(
                            src_ref=cm_w2_pad.at[_c],
                            dst_ref=w2_buf.at[slot_b],
                            sem=w2_sem.at[slot_b],
                        ).start()

                @pl.when(eid < borrowed_slot)
                def _from_home():
                    pltpu.make_async_copy(
                        src_ref=w1_hbm.at[eid],
                        dst_ref=w1_buf.at[slot_b],
                        sem=w1_sem.at[slot_b],
                    ).start()
                    pltpu.make_async_copy(
                        src_ref=w2_hbm.at[eid],
                        dst_ref=w2_buf.at[slot_b],
                        sem=w2_sem.at[slot_b],
                    ).start()

            if split_mode:

                @pl.when(e0 >= borrowed_slot)
                def _gm0_borrowed():
                    _start_w(0, e0)
            else:
                _start_w(0, e0)

            for gm, eid in ((0, e0), (1, e1)):
                pltpu.make_async_copy(
                    src_ref=x_vmem.at[gm], dst_ref=x_vmem.at[gm], sem=g_sem.at[gm]
                ).wait()
                pltpu.make_async_copy(
                    src_ref=w1_buf.at[gm], dst_ref=w1_buf.at[gm], sem=w1_sem.at[gm]
                ).wait()
                y1 = x_vmem[gm] + w1_buf[gm]
                if gm == 0:
                    _issue_def_send()
                    _start_w(1, e1)
                pltpu.make_async_copy(
                    src_ref=w2_buf.at[gm], dst_ref=w2_buf.at[gm], sem=w2_sem.at[gm]
                ).wait()
                out_hbm[gm] = y1 + w2_buf[gm]

            for ci in range(2):
                pltpu.make_async_remote_copy(
                    src_ref=w1_hbm.at[ci],
                    dst_ref=cm_w1_pad.at[ci],
                    send_sem=cm_send_sem.at[2 * ci + 0],
                    recv_sem=cm_recv_sem.at[2 * ci + 0],
                    device_id={"ep": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
                pltpu.make_async_remote_copy(
                    src_ref=w2_hbm.at[ci],
                    dst_ref=cm_w2_pad.at[ci],
                    send_sem=cm_send_sem.at[2 * ci + 1],
                    recv_sem=cm_recv_sem.at[2 * ci + 1],
                    device_id={"ep": partner},
                    device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
                _wait_cm_w1(ci)
                _wait_cm_w2(ci)

            pl.semaphore_signal(
                bar_sem.at[0],
                1,
                device_id={"ep": partner},
                device_id_type=pl.DeviceIdType.MESH,
            )
            pl.semaphore_wait(bar_sem.at[0], 1)

        def _shard(x_s, w1_s, w2_s):
            out, _, _ = pl.pallas_call(
                _ker,
                out_shape=(
                    jax.ShapeDtypeStruct((2, 16, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((2, 16, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((2, 16, 128), jnp.bfloat16),
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 3,
                    out_specs=(
                        pl.BlockSpec(memory_space=pltpu.VMEM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                    ),
                    scratch_shapes=[
                        pltpu.VMEM((2, 16, 128), jnp.bfloat16),
                        pltpu.VMEM((2, 16, 128), jnp.bfloat16),
                        pltpu.VMEM((2, 16, 128), jnp.bfloat16),
                        pltpu.SemaphoreType.DMA((2,)),
                        pltpu.SemaphoreType.DMA((2,)),
                        pltpu.SemaphoreType.DMA((2,)),
                        pltpu.SemaphoreType.DMA((4,)),
                        pltpu.SemaphoreType.DMA((4,)),
                        pltpu.SMEM((cm_flag_len,), jnp.int32),
                        pltpu.SemaphoreType.REGULAR((1,)),
                    ],
                ),
                interpret=pltpu.InterpretParams(
                    detect_races=True, uninitialized_memory="nan"
                ),
            )(x_s[0], w1_s[0], w2_s[0])
            return out[None, ...]

        res = jax.jit(
            jax.shard_map(
                _shard,
                mesh=mesh,
                in_specs=(P("ep"), P("ep"), P("ep")),
                out_specs=P("ep"),
                check_vma=False,
            )
        )(x_data, w1_data, w2_data)
        return np.asarray(res, dtype=np.float32)

    ref_g2a4 = _run(split_mode=False)
    got_g2a6 = _run(split_mode=True)
    assert np.isfinite(got_g2a6).all()
    np.testing.assert_array_equal(got_g2a6, ref_g2a4)


@pytest.mark.parametrize(
    "ep_size,num_tok,expect_single_deq",
    [(8, 16384, False), (16, 16384, False), (16, 2048, True)],
)
def test_w9_pql1_g2a6_2x2x1_and_2x2x2_mosaic_lowering(
    ep_size: int, num_tok: int, expect_single_deq: bool
) -> None:
    """Verify TPU7x Mosaic lowering for pql1 + g2a6 + cm1s2d across EP=8 and EP=16 (1-half & 2-half)."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices()) < ep_size:
        pytest.skip(f"Requires >={ep_size} virtual/physical JAX devices")

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    mesh = Mesh(np.array(jax.devices()[:ep_size]), ("model",))
    e_loc = 256 // ep_size + 2
    with (
        patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
        patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
                "SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "2",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "0",
                "SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
                "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
                "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
                "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
                "SGLANG_TPU_MUSE_KERNEL_GMM_PREFILL_ROW_BUCKETS": "4",
                "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
                "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
            },
        ),
    ):
        jax.clear_caches()

        def _wrap(a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus):
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
                    d_fp8_unpool=True,
                    in_shard_map=True,
                    b2_deq0_new_buffer=True,
                    next_in_gather_rows=min(512, num_tok // ep_size),
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
                    P("model"),
                    P("model"),
                    P("model"),
                    P("model"),
                    P(),
                    P("model"),
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
            )(a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus, pfw, aa, ba, pfw)

        hlo = (
            jax.jit(_wrap)
            .trace(
                jax.ShapeDtypeStruct((num_tok, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((num_tok, 8192), jnp.float32),
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
                jax.ShapeDtypeStruct((256, 8, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((256, 8, 4096), jnp.bfloat16),
                jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                jax.ShapeDtypeStruct((ep_size,), jnp.int32),
                jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert f"gmm_v2_fused_rs-E_{e_loc}-M_{num_tok * 8}" in hlo
        assert "-g2a6" in hlo and "-pql1" in hlo and "-cm1s2d" in hlo
        expected_deq = (
            "pallas_deq_single_fp8_gmm"
            if expect_single_deq
            else "pallas_deq_both_fp8_gmm"
        )
        assert expected_deq in hlo
        jax.clear_caches()


def test_w9_b1_b2_emit_fp8_gmm_interpret_zero_ulp():
    """Verify emit_fp8_gmm=True (1-pass & 2-half) matches reference at 0-ULP on CPU interpret."""
    from unittest.mock import patch

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_b2 import (
        pallas_fused_b1_b2_wpool_quant,
    )

    orig_pallas_call = pl.pallas_call

    def _interp_pallas_call(*args, **kwargs):
        kwargs["interpret"] = pltpu.InterpretParams(uninitialized_memory="nan")
        return orig_pallas_call(*args, **kwargs)

    rng = np.random.default_rng(20261124)
    t_loc = 256
    h_dim = 8192
    he_dim = 4096
    n_exp = 256
    attn_s = jnp.asarray(rng.standard_normal((t_loc, h_dim), np.float32) * 0.3, jnp.bfloat16)
    stream_s = jnp.asarray(rng.standard_normal((t_loc, h_dim), np.float32) * 0.3, jnp.float32)
    a_attn = jnp.full((h_dim,), 1.05, jnp.float32)
    b_attn = jnp.full((h_dim,), 0.85, jnp.float32)
    pffn_w = jnp.asarray(1.0 + 0.05 * rng.standard_normal(h_dim, np.float32), jnp.bfloat16)
    w_r = jnp.asarray(rng.standard_normal((h_dim, n_exp), np.float32) * 0.05, jnp.float32)
    bias = jnp.zeros((n_exp,), jnp.float32)
    wp_fp8 = jnp.asarray(rng.standard_normal((h_dim, he_dim), np.float32) * 0.05, jnp.float8_e4m3fn)
    wp_sc = jnp.full((he_dim,), 0.05, jnp.float32)
    pe_w = jnp.asarray(1.0 + 0.05 * rng.standard_normal(he_dim, np.float32), jnp.bfloat16)

    with patch("jax.experimental.pallas.pallas_call", side_effect=_interp_pallas_call):
        # 1-pass (pql4) vs stock emit_fp8_gmm=False
        i8_ref, smid_ref, tk_id_ref, tk_w_ref = pallas_fused_b1_b2_wpool_quant(
            attn_s, stream_s, a_attn, b_attn, pffn_w, w_r, bias, wp_fp8, wp_sc, pe_w,
            router_mode="bf16_hilo", fp8_wpool=True, half_idx=None, topk=8, emit_fp8_gmm=False,
        )
        fp8_1p, sc_1p, smid_1p, tk_id_1p, tk_w_1p = pallas_fused_b1_b2_wpool_quant(
            attn_s, stream_s, a_attn, b_attn, pffn_w, w_r, bias, wp_fp8, wp_sc, pe_w,
            router_mode="bf16_hilo", fp8_wpool=True, half_idx=None, topk=8, emit_fp8_gmm=True,
        )
        # 2-half (pql5, exercising stream_mid_prev HBM output buffer aliasing {8: 2})
        fp8_h0, sc_h0, smid_h0, tk_id_h0, tk_w_h0 = pallas_fused_b1_b2_wpool_quant(
            attn_s, stream_s, a_attn, b_attn, pffn_w, w_r, bias, wp_fp8, wp_sc, pe_w,
            router_mode="bf16_hilo", fp8_wpool=True, half_idx=0, full_stream_mid=True,
            topk=8, emit_fp8_gmm=True,
        )
        fp8_h1, sc_h1, smid_2h, tk_id_h1, tk_w_h1 = pallas_fused_b1_b2_wpool_quant(
            attn_s, stream_s, a_attn, b_attn, pffn_w, w_r, bias, wp_fp8, wp_sc, pe_w,
            router_mode="bf16_hilo", fp8_wpool=True, half_idx=1, stream_mid_prev=smid_h0,
            topk=8, emit_fp8_gmm=True,
        )

    assert fp8_1p.shape == (t_loc, he_dim // 128, 128) and fp8_1p.dtype == jnp.float8_e4m3fn
    assert sc_1p.shape == (t_loc,) and sc_1p.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(smid_1p), np.asarray(smid_ref))
    np.testing.assert_array_equal(np.asarray(tk_id_1p), np.asarray(tk_id_ref))
    np.testing.assert_array_equal(np.asarray(tk_w_1p), np.asarray(tk_w_ref))

    # 2-half pql5 must match 1-pass pql4 bit-for-bit (0-ULP)!
    fp8_2h = jnp.concatenate([fp8_h0, fp8_h1], axis=0)
    sc_2h = jnp.concatenate([sc_h0, sc_h1], axis=0)
    np.testing.assert_array_equal(
        np.asarray(jax.lax.bitcast_convert_type(fp8_2h, jnp.uint8)),
        np.asarray(jax.lax.bitcast_convert_type(fp8_1p, jnp.uint8)),
    )
    np.testing.assert_array_equal(np.asarray(sc_2h), np.asarray(sc_1p))
    np.testing.assert_array_equal(np.asarray(smid_2h), np.asarray(smid_ref))
    np.testing.assert_array_equal(
        np.asarray(jnp.concatenate([tk_id_h0, tk_id_h1], axis=0)), np.asarray(tk_id_ref)
    )
    np.testing.assert_array_equal(
        np.asarray(jnp.concatenate([tk_w_h0, tk_w_h1], axis=0)), np.asarray(tk_w_ref)
    )

    # Dequantized FP8 hidden states must closely match dequantized int8 reference (>0.999 cos_sim).
    sc_i8_u16 = (
        i8_ref[:, he_dim : he_dim + 1].astype(jnp.uint8).astype(jnp.uint16)
        | (i8_ref[:, he_dim + 64 : he_dim + 65].astype(jnp.uint8).astype(jnp.uint16) << 8)
    )
    sc_i8 = jax.lax.bitcast_convert_type(sc_i8_u16, jnp.bfloat16).astype(jnp.float32)
    deq_i8 = i8_ref[:, :he_dim].astype(jnp.float32) * sc_i8
    deq_fp8 = fp8_1p.reshape(t_loc, he_dim).astype(jnp.float32) * sc_1p[:, None]
    assert np.isfinite(np.asarray(deq_fp8)).all()
    cos = float(
        np.dot(np.asarray(deq_fp8).ravel(), np.asarray(deq_i8).ravel())
        / (np.linalg.norm(np.asarray(deq_fp8)) * np.linalg.norm(np.asarray(deq_i8)) + 1e-30)
    )
    assert cos > 0.999, f"Expected cos > 0.999 between direct FP8 and int8 B2 outputs, got {cos}"


@pytest.mark.parametrize(
    "ep_size,num_tok,pq_mode",
    [
        (8, 16384, "4"),
        (8, 16384, "5"),
        (16, 16384, "4"),
        (16, 16384, "5"),
    ],
)
def test_w9_pql4_pql5_zero_deq_pipeline_lowering(ep_size: int, num_tok: int, pq_mode: str):
    """Verify pql4 (1-pass) and pql5 (2-half) lower cleanly on 2x2x1 & 2x2x2 with ZERO post-AG deq kernel."""
    from unittest.mock import patch

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices()) < ep_size:
        pytest.skip(f"Needs >={ep_size} JAX devices (use XLA_FLAGS=--xla_force_host_platform_device_count=16)")

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    mesh = Mesh(np.array(jax.devices()[:ep_size]), ("model",))
    e_loc = 256 // ep_size + 2
    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
                "SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "2",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "0",
                "SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
                "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
                "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
                "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
                "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
                "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": pq_mode,
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
            },
        ),
    ):
        jax.clear_caches()

        def _wrap(a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus):
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
                    d_fp8_unpool=True,
                    in_shard_map=True,
                    b2_deq0_new_buffer=True,
                    next_in_gather_rows=0,
                    next_in_gather_static_fp8=False,
                ),
                mesh=mesh,
                in_specs=(
                    P("model", None), P("model", None), P(), P(), P(), P(), P(), P(), P(), P(),
                    P("model"), P("model"), P("model"), P("model"), P(), P("model"),
                    P(), P(), P(), P(), P(), P(),
                ),
                out_specs=(
                    P("model", None), P("model", None), P("model", None), P("model", None),
                ),
                check_vma=False,
            )(a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus, pfw, aa, ba, pfw)

        hlo = (
            jax.jit(_wrap)
            .trace(
                jax.ShapeDtypeStruct((num_tok, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((num_tok, 8192), jnp.float32),
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
                jax.ShapeDtypeStruct((256, 8, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((256, 8, 4096), jnp.bfloat16),
                jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                jax.ShapeDtypeStruct((ep_size,), jnp.int32),
                jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert f"gmm_v2_fused_rs-E_{e_loc}-M_{num_tok * 8}" in hlo
        assert f"-pql{pq_mode}" in hlo and "-g2a6" in hlo and "-cm1s2d" in hlo
        assert "_outfp8" in hlo
        assert "pallas_deq_both_fp8_gmm" not in hlo
        assert "pallas_deq_single_fp8_gmm" not in hlo
        assert "pallas_deq_both_gmm" not in hlo
        assert "pallas_deq_single_gmm" not in hlo
        if ep_size == 8:
            assert "-u16w" not in hlo
        jax.clear_caches()


def test_w9_cand4_eo3_eo4_with_pql4_interpret_zero_ulp_and_lowering():
    """Verify run_eo3_expert_batch + has_ext_pq_lhs (pql4) matches eo4 at 0.0-ULP on log-normal scales and lowers cleanly."""
    from unittest.mock import patch

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        fused_rs_smem_rows,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_v2_gather_scatter import MetadataRef
    from sglang_tpu.moe.pallas.fused_rs_muse.w1_fp8_expert_cache import (
        compute_eo3_batch_count,
        run_eo3_expert_batch,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    # 1. Verify i32w98k SMEM windowing at M=131,072 under PREQUANT_LHS_FP8>=1:
    with patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "4"}):
        assert fused_rs_smem_rows(131_072, 8, False) == 98_304
    with patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "0"}):
        assert fused_rs_smem_rows(131_072, 8, False) == 131_072

    # 2. CPU interpret=True: eo4+pql4 must match eo4+bf16 at 0.0-ULP when fed identical quantized FP8 LHS!
    rng = np.random.default_rng(20261125)
    num_gm = 4
    e_ids = np.array([0, 1, 1, 1, 0], dtype=np.int32)
    tile_m, k1, n1_half, k2, n2, rhs_qbs = 256, 2048, 512, 1024, 512, 256
    m_offs = np.arange(num_gm + 1, dtype=np.int32) * tile_m
    num_e = 2
    n_tok = num_gm * tile_m
    lhs_idx = np.arange(n_tok, dtype=np.int32)

    x_bf16 = jnp.asarray(
        rng.standard_normal((n_tok, k1 // 128, 128)).astype(np.float32) * 0.4,
        jnp.bfloat16,
    )
    x_2d = x_bf16.reshape(n_tok, k1)
    amax = jnp.max(jnp.abs(x_2d), axis=1, keepdims=True).astype(jnp.float32)
    inv_s = jnp.where(amax == 0.0, 0.0, jnp.float32(448.0) / amax).astype(jnp.bfloat16)
    row_sc = jnp.where(amax == 0.0, 0.0, amax * jnp.float32(1.0 / 448.0)).reshape(-1)
    x_fp8 = (x_2d * inv_s).astype(jnp.float8_e4m3fn).reshape(n_tok, k1 // 128, 128)

    w1_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(num_e, 2, k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    s1_bf16 = jnp.asarray(
        np.exp(rng.normal(-2.0, 0.35, size=(num_e, 2, k1 // rhs_qbs, 1, n1_half * 2))).astype(np.float32),
        jnp.bfloat16,
    )
    w2_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(num_e, 2, k2 // 8, n2), dtype=np.uint32)
    )
    s2_bf16 = jnp.asarray(
        np.exp(rng.normal(-2.0, 0.35, size=(num_e, 2, k2 // rhs_qbs, 1, n2))).astype(np.float32),
        jnp.bfloat16,
    )
    pnw_8 = jnp.broadcast_to(
        jnp.asarray(1.0 + 0.08 * rng.standard_normal((n2 * 2,), np.float32), jnp.bfloat16)[None, :],
        (8, n2 * 2),
    )
    gs_g = jnp.asarray([0.9, 1.1], dtype=jnp.float32)
    gs_u = jnp.asarray([1.05, 0.95], dtype=jnp.float32)
    gs_w2 = jnp.asarray([0.97, 1.03], dtype=jnp.float32)

    def _run_eo_interp(*, use_ext_pq: bool, direct_fp8: bool) -> np.ndarray:
        x_in = x_fp8.reshape(num_gm, tile_m, k1 // 128, 128) if use_ext_pq else x_bf16.reshape(num_gm, tile_m, k1 // 128, 128)

        def _ker(
            x_r, sc_r, idx_r, w1_r, s1_r, w2_r, s2_r, pnw_r, eid_r, moff_r, gsg_r, gsu_r, gsw2_r,
            out_r, g_lhs_2x, w1_b, s1_b, w2_b, s2_b, stg_4x, wh_r, wc_r, lfp8_5x, lsc_5x, buf_5x,
            gmid_r, fg_r, fm_r, gsem_r,
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
                bn = compute_eo3_batch_count(gm_id, jnp.int32(num_gm), meta, max_batch=5)
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
                        lhs_indices_ref=idx_r,
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
                        static_fp8_rs_scale=0.0,
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
                        has_ext_pq_lhs=use_ext_pq,
                        lhs_row_scales_ref=sc_r,
                        pq_lhs_mode=(4 if use_ext_pq else 0),
                    )

                return nxt_skip

            jax.lax.fori_loop(0, num_gm, _body, jnp.int32(0))

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            return np.asarray(
                pl.pallas_call(
                    _ker,
                    out_shape=jax.ShapeDtypeStruct((num_gm, tile_m, n2 * 2), jnp.bfloat16),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)]
                        + [pl.BlockSpec(memory_space=pltpu.SMEM)] * 2
                        + [pl.BlockSpec(memory_space=pltpu.VMEM)] * 5
                        + [pl.BlockSpec(memory_space=pltpu.SMEM)] * 5,
                        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                        scratch_shapes=[
                            pltpu.VMEM((2, tile_m, k1 // 128, 128), x_in.dtype),
                            pltpu.VMEM((2, k1 // 8, n1_half * 2), jnp.uint32),
                            pltpu.VMEM((2, k1 // rhs_qbs, 1, n1_half * 2), jnp.bfloat16),
                            pltpu.VMEM((2, k2 // 8, n2), jnp.uint32),
                            pltpu.VMEM((2, k2 // rhs_qbs, 1, n2), jnp.bfloat16),
                            pltpu.VMEM((4, tile_m, (n2 * 2) // 128, 128), jnp.bfloat16),
                            pltpu.VMEM((max(k1, k2), max(n1_half, n2)), jnp.float8_e4m3fn),
                            pltpu.VMEM((8, 128 if direct_fp8 else max(n1_half, n2)), jnp.float32),
                            pltpu.VMEM((5, tile_m, max(k1, k2)), jnp.float8_e4m3fn),
                            pltpu.VMEM((5, tile_m, 128), jnp.float32),
                            pltpu.VMEM((5, tile_m, n1_half * 2), jnp.bfloat16),
                            pltpu.SMEM((1,), jnp.int32),
                            pltpu.SMEM((1,), jnp.int32),
                            pltpu.SMEM((2,), jnp.int32),
                            pltpu.SemaphoreType.DMA((2,)),
                        ],
                    ),
                    interpret=pltpu.InterpretParams(uninitialized_memory="nan"),
                )(
                    x_in, row_sc, jnp.asarray(lhs_idx), w1_u32, s1_bf16, w2_u32, s2_bf16,
                    pnw_8, jnp.asarray(e_ids), jnp.asarray(m_offs), gs_g, gs_u, gs_w2,
                ),
                dtype=np.float32,
            )

    out_eo4_bf16 = _run_eo_interp(use_ext_pq=False, direct_fp8=True)
    out_eo4_pql4 = _run_eo_interp(use_ext_pq=True, direct_fp8=True)
    out_eo3_bf16 = _run_eo_interp(use_ext_pq=False, direct_fp8=False)
    out_eo3_pql4 = _run_eo_interp(use_ext_pq=True, direct_fp8=False)

    # 0.0-ULP bit-exact match between prequantized FP8 LHS (pql4) and in-kernel FP8 LHS!
    np.testing.assert_array_equal(out_eo4_pql4, out_eo4_bf16)
    np.testing.assert_array_equal(out_eo3_pql4, out_eo3_bf16)

    # Verify eo3 vs eo4 cosine > 0.999 on log-normal scales
    cos_eo3 = float(
        np.dot(out_eo3_pql4.ravel(), out_eo4_pql4.ravel())
        / (np.linalg.norm(out_eo3_pql4) * np.linalg.norm(out_eo4_pql4) + 1e-30)
    )
    assert cos_eo3 > 0.997, f"Expected cos > 0.997 between eo3+pql4 and eo4+pql4, got {cos_eo3}"

    if len(jax.devices()) < 8:
        pytest.skip("8-device lowering check needs XLA_FLAGS=--xla_force_host_platform_device_count=8")

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    for eo_mode in ("3", "4"):
        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
            patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
                    "SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "3",
                    "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                    "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "0",
                    "SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
                    "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
                    "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "4",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8": eo_mode,
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
                    "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
                },
            ),
        ):
            jax.clear_caches()

            def _wrap(a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus):
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
                        d_fp8_unpool=True,
                        in_shard_map=True,
                        b2_deq0_new_buffer=True,
                        next_in_gather_rows=0,
                        next_in_gather_static_fp8=False,
                    ),
                    mesh=mesh,
                    in_specs=(
                        P("model", None), P("model", None), P(), P(), P(), P(), P(), P(), P(), P(),
                        P("model"), P("model"), P("model"), P("model"), P(), P("model"),
                        P(), P(), P(), P(), P(), P(),
                    ),
                    out_specs=(P("model", None), P("model", None), P("model", None), P("model", None)),
                    check_vma=False,
                )(a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus, pfw, aa, ba, pfw)

            hlo = (
                jax.jit(_wrap)
                .trace(
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
                    jax.ShapeDtypeStruct((256, 8, 8192), jnp.bfloat16),
                    jax.ShapeDtypeStruct((256, 8, 4096), jnp.bfloat16),
                    jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8,), jnp.int32),
                    jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                )
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )
            assert f"-pql4" in hlo and f"-eo{eo_mode}" in hlo
            assert "-u16w" not in hlo
            assert "_outfp8" in hlo and "xor1_xytorus_ag" in hlo
            assert "pallas_deq_both" not in hlo
            jax.clear_caches()


@pytest.mark.parametrize("num_w2_bufs", [1, 2])
def test_w9_cand5_wla2_pingpong_nan_interpret_zero_ulp_and_lowering(num_w2_bufs: int):
    """Verify wla2 2-buffer expert ping-pong + 2-slot chipmate (_next_act_grp_off=5) under
    InterpretParams(detect_races=True, uninitialized_memory='nan') (0 NaNs, 0 races, 0.0-ULP)
    and TPU7x EP=8 / EP=16 Mosaic lowering."""
    from unittest.mock import patch

    import jax
    import jax.numpy as jnp
    import numpy as np
    import pytest
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    if len(jax.devices()) < 2:
        pytest.skip("Requires >=2 virtual/physical JAX devices")

    rng = np.random.default_rng(20261127 + num_w2_bufs)
    w1_data = jnp.asarray(rng.standard_normal((2, 3, 16, 128), np.float32) * 0.4, jnp.bfloat16)
    w2_data = jnp.asarray(rng.standard_normal((2, 3, 16, 128), np.float32) * 0.4, jnp.bfloat16)
    x_data = jnp.asarray(rng.standard_normal((2, 5, 16, 128), np.float32) * 0.4, jnp.bfloat16)
    # 5 gm tiles: E_0 (2 tiles: gm=0,1), E_2 (1 tile: gm=2), borrowed E_3 (2 tiles: gm=3,4)
    gm_eids = jnp.asarray([0, 0, 2, 3, 3], jnp.int32)
    # max_num_gm_ref: [max_gm(5), cm0_send(0), cm0_recv(1), cm1_send(1), cm1_recv(1)] + _next_act_grp[0..4]
    # Active experts are {0, 2, 3} -> next_act_grp = [2, 2, 3, -1, -1] at offset 5!
    max_gm_meta = jnp.asarray([5, 0, 1, 1, 1, 2, 2, 3, -1, -1], jnp.int32)
    mesh2 = Mesh(np.array(jax.devices()[:2]), ("ep",))

    def _run_wla(use_wla2: bool) -> np.ndarray:
        def _ker(
            meta_r, eid_r, x_hbm, w1_hbm, w2_hbm,
            out_hbm, cm_w1_pad, cm_w2_pad,
            x_vmem, w1_buf, w2_buf,
            g_sem, w1_sem, w2_sem, cm_send_sem, cm_recv_sem, cm_awaited, st_smem, bar_sem,
        ):
            my_id = jax.lax.axis_index("ep")
            partner = jax.lax.bitwise_xor(my_id, jnp.int32(1))
            borrowed_slot = jnp.int32(3)
            next_act_off = 5
            for i in range(5):
                cm_awaited[i] = jnp.int32(0)
            st_smem[0] = jnp.int32(0)
            st_smem[1] = jnp.int32(0)

            pl.semaphore_signal(bar_sem.at[0], 1, device_id={"ep": partner}, device_id_type=pl.DeviceIdType.MESH)
            e0 = eid_r[0]
            pltpu.make_async_copy(src_ref=x_hbm.at[0], dst_ref=x_vmem.at[0], sem=g_sem.at[0]).start()

            @pl.when(e0 < borrowed_slot)
            def _pro_w():
                pltpu.make_async_copy(src_ref=w1_hbm.at[e0], dst_ref=w1_buf.at[0], sem=w1_sem.at[0]).start()
                pltpu.make_async_copy(src_ref=w2_hbm.at[e0], dst_ref=w2_buf.at[0], sem=w2_sem.at[0]).start()

            pltpu.make_async_copy(src_ref=x_hbm.at[1], dst_ref=x_vmem.at[1], sem=g_sem.at[1]).start()
            pl.semaphore_wait(bar_sem.at[0], 1)

            def _def_send():
                @pl.when(cm_awaited[4] == 0)
                def _do():
                    for ci in range(2):
                        pltpu.make_async_remote_copy(
                            src_ref=w1_hbm.at[ci], dst_ref=cm_w1_pad.at[ci],
                            send_sem=cm_send_sem.at[2 * ci], recv_sem=cm_recv_sem.at[2 * ci],
                            device_id={"ep": partner}, device_id_type=pl.DeviceIdType.MESH,
                        ).start()
                        pltpu.make_async_remote_copy(
                            src_ref=w2_hbm.at[ci], dst_ref=cm_w2_pad.at[ci],
                            send_sem=cm_send_sem.at[2 * ci + 1], recv_sem=cm_recv_sem.at[2 * ci + 1],
                            device_id={"ep": partner}, device_id_type=pl.DeviceIdType.MESH,
                        ).start()
                    cm_awaited[4] = jnp.int32(1)

            def _wait_cm_w1(ci: int):
                @pl.when(cm_awaited[ci] == 0)
                def _():
                    pltpu.make_async_copy(src_ref=cm_w1_pad.at[ci], dst_ref=cm_w1_pad.at[ci], sem=cm_recv_sem.at[2 * ci]).wait()
                    cm_awaited[ci] = jnp.int32(1)

            def _wait_cm_w2(ci: int):
                @pl.when(cm_awaited[2 + ci] == 0)
                def _():
                    pltpu.make_async_copy(src_ref=cm_w2_pad.at[ci], dst_ref=cm_w2_pad.at[ci], sem=cm_recv_sem.at[2 * ci + 1]).wait()
                    cm_awaited[2 + ci] = jnp.int32(1)

            def _start_w1(sl, eid):
                for ci in range(2):
                    @pl.when(eid == borrowed_slot + ci)
                    def _cm(_c=ci):
                        _wait_cm_w1(_c)
                        pltpu.make_async_copy(src_ref=cm_w1_pad.at[_c], dst_ref=w1_buf.at[sl], sem=w1_sem.at[sl]).start()

                @pl.when(eid < borrowed_slot)
                def _hm():
                    pltpu.make_async_copy(src_ref=w1_hbm.at[eid], dst_ref=w1_buf.at[sl], sem=w1_sem.at[sl]).start()

            def _start_w2(sl, eid):
                for ci in range(2):
                    @pl.when(eid == borrowed_slot + ci)
                    def _cm(_c=ci):
                        _wait_cm_w2(_c)
                        pltpu.make_async_copy(src_ref=cm_w2_pad.at[_c], dst_ref=w2_buf.at[sl], sem=w2_sem.at[sl]).start()

                @pl.when(eid < borrowed_slot)
                def _hm():
                    pltpu.make_async_copy(src_ref=w2_hbm.at[eid], dst_ref=w2_buf.at[sl], sem=w2_sem.at[sl]).start()

            def _body(gm_id, carry):
                sem_id = gm_id % 2
                eid = eid_r[gm_id]
                prev_e = eid_r[jnp.maximum(gm_id - 1, 0)]
                is_new = jnp.logical_or(gm_id == 0, prev_e != eid)
                has_nxt = gm_id + 1 < 5
                nxt_e = eid_r[jnp.minimum(gm_id + 1, 4)]

                @pl.when(jnp.logical_and(gm_id == 0, eid >= borrowed_slot))
                def _():
                    _def_send()
                    _start_w1(0, eid)
                    _start_w2(0, eid)

                if use_wla2:
                    @pl.when(jnp.logical_and(gm_id > 0, is_new))
                    def _adv():
                        st_smem[0] = jnp.int32(1) - st_smem[0]
                        st_smem[1] = jnp.int32(0)

                    w_base = st_smem[0]
                    sl_la = jnp.int32(1) - w_base
                    e_after = meta_r[next_act_off + eid]

                    @pl.when(gm_id == 0)
                    def _():
                        _def_send()

                    @pl.when(jnp.logical_and(e_after >= 0, st_smem[1] == 0))
                    def _la():
                        _start_w1(sl_la, e_after)
                        if num_w2_bufs >= 2:
                            _start_w2(sl_la, e_after)
                        st_smem[1] = jnp.int32(1)
                else:
                    w_base = jnp.int32(0)

                pltpu.make_async_copy(src_ref=x_vmem.at[sem_id], dst_ref=x_vmem.at[sem_id], sem=g_sem.at[sem_id]).wait()
                b1 = w_base if use_wla2 else jnp.int32(0)

                @pl.when(is_new)
                def _():
                    pltpu.make_async_copy(src_ref=w1_buf.at[b1], dst_ref=w1_buf.at[b1], sem=w1_sem.at[b1]).wait()

                y1 = x_vmem[sem_id] + w1_buf[b1]

                @pl.when(gm_id + 2 < 5)
                def _():
                    pltpu.make_async_copy(src_ref=x_hbm.at[gm_id + 2], dst_ref=x_vmem.at[sem_id], sem=g_sem.at[sem_id]).start()

                if not use_wla2:
                    @pl.when(gm_id == 0)
                    def _():
                        _def_send()

                    @pl.when(jnp.logical_and(has_nxt, nxt_e != eid))
                    def _():
                        _start_w1(0, nxt_e)

                b2 = (w_base if num_w2_bufs >= 2 else jnp.int32(0)) if use_wla2 else jnp.int32(0)

                @pl.when(is_new)
                def _():
                    pltpu.make_async_copy(src_ref=w2_buf.at[b2], dst_ref=w2_buf.at[b2], sem=w2_sem.at[b2]).wait()

                out_hbm[gm_id] = y1 + w2_buf[b2]
                if not (use_wla2 and num_w2_bufs >= 2):
                    @pl.when(jnp.logical_and(has_nxt, nxt_e != eid))
                    def _():
                        _start_w2(0, nxt_e)
                return carry

            jax.lax.fori_loop(0, 5, _body, jnp.int32(0))
            for ci in range(2):
                pltpu.make_async_remote_copy(
                    src_ref=w1_hbm.at[ci], dst_ref=cm_w1_pad.at[ci],
                    send_sem=cm_send_sem.at[2 * ci], recv_sem=cm_recv_sem.at[2 * ci],
                    device_id={"ep": partner}, device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
                pltpu.make_async_remote_copy(
                    src_ref=w2_hbm.at[ci], dst_ref=cm_w2_pad.at[ci],
                    send_sem=cm_send_sem.at[2 * ci + 1], recv_sem=cm_recv_sem.at[2 * ci + 1],
                    device_id={"ep": partner}, device_id_type=pl.DeviceIdType.MESH,
                ).wait_send()
                _wait_cm_w1(ci)
                _wait_cm_w2(ci)
            pl.semaphore_signal(bar_sem.at[0], 1, device_id={"ep": partner}, device_id_type=pl.DeviceIdType.MESH)
            pl.semaphore_wait(bar_sem.at[0], 1)

        def _shard(m_s, e_s, x_s, w1_s, w2_s):
            out, _, _ = pl.pallas_call(
                _ker,
                out_shape=(
                    jax.ShapeDtypeStruct((5, 16, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((2, 16, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((2, 16, 128), jnp.bfloat16),
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=2,
                    in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * 3,
                    out_specs=(
                        pl.BlockSpec(memory_space=pltpu.VMEM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                    ),
                    scratch_shapes=[
                        pltpu.VMEM((2, 16, 128), jnp.bfloat16),
                        pltpu.VMEM((2, 16, 128), jnp.bfloat16),
                        pltpu.VMEM((num_w2_bufs, 16, 128), jnp.bfloat16),
                        pltpu.SemaphoreType.DMA((2,)),
                        pltpu.SemaphoreType.DMA((2,)),
                        pltpu.SemaphoreType.DMA((num_w2_bufs,)),
                        pltpu.SemaphoreType.DMA((4,)),
                        pltpu.SemaphoreType.DMA((4,)),
                        pltpu.SMEM((5,), jnp.int32),
                        pltpu.SMEM((2,), jnp.int32),
                        pltpu.SemaphoreType.REGULAR((1,)),
                    ],
                ),
                interpret=pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan"),
            )(m_s, e_s, x_s[0], w1_s[0], w2_s[0])
            return out[None, ...]

        return np.asarray(
            jax.jit(
                jax.shard_map(
                    _shard,
                    mesh=mesh2,
                    in_specs=(P(), P(), P("ep"), P("ep"), P("ep")),
                    out_specs=P("ep"),
                    check_vma=False,
                )
            )(max_gm_meta, gm_eids, x_data, w1_data, w2_data),
            dtype=np.float32,
        )

    ref_wla0 = _run_wla(False)
    got_wla2 = _run_wla(True)
    assert np.isfinite(got_wla2).all()
    np.testing.assert_array_equal(got_wla2, ref_wla0)

    if len(jax.devices()) < 8:
        pytest.skip("8-rank HLO check needs >=8 virtual/physical JAX devices")

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    tiles_str = f"{256 if num_w2_bufs == 1 else 128},4096,4096,4096,4096,2,{num_w2_bufs}"
    mesh8 = Mesh(np.array(jax.devices()[:8]), ("model",))
    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
                "SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "0",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "0",
                "SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_REBALANCE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_SLOTS": "2",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_CHIPMATE_MODE": "split",
                "SGLANG_TPU_MUSE_KERNEL_CHIPMATE_DEFER_SEND": "1",
                "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
                "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
                "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
                "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "4",
                "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "2",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": tiles_str,
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "512",
            },
        ),
    ):
        jax.clear_caches()

        def _wrap(a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus):
            return jax.shard_map(
                lambda *xs: fused_moe_full_bridge_and_ffn(
                    *xs,
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
                    next_in_gather_rows=0,
                    next_in_gather_static_fp8=False,
                ),
                mesh=mesh8,
                in_specs=(
                    P("model", None), P("model", None), P(), P(), P(), P(), P(), P(), P(), P(),
                    P("model"), P("model"), P("model"), P("model"), P(), P("model"),
                    P(), P(), P(), P(), P(), P(),
                ),
                out_specs=(P("model", None), P("model", None), P("model", None), P("model", None)),
                check_vma=False,
            )(a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus, pfw, aa, ba, pfw)

        hlo = (
            jax.jit(_wrap)
            .trace(
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
                jax.ShapeDtypeStruct((256, 8, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((256, 8, 4096), jnp.bfloat16),
                jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                jax.ShapeDtypeStruct((8,), jnp.int32),
                jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert "-wla2" in hlo and "-pql4" in hlo and "-g2a6" in hlo
        jax.clear_caches()


def test_w9_cand6_m16k_tm128_stepdown_and_rb4_buckets():
    """Verify PREFILL_GMM_TILES='256,...' steps down to tm=128 at M=16K..24K while keeping tm=256 at M>=32K."""
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        _PREFILL_TILES_MIN_SIZE_M,
        _PREFILL_TM256_MIN_SIZE_M,
        _select_fused_rs_block_sizes,
    )
    from sglang_tpu.pallas_export import tpu_info_for_cpu_export

    assert _PREFILL_TILES_MIN_SIZE_M == 16384
    assert _PREFILL_TM256_MIN_SIZE_M == 32768

    def _query(m: int) -> tuple[int, ...]:
        bs = _select_fused_rs_block_sizes(
            size_m=m,
            size_k1=4096,
            size_n1=8192,
            size_k2=4096,
            size_n2=4096,
            size_group=32,
            size_lhs_group=32,
            ep_size=8,
            out_dtype=jnp.bfloat16,
            w1_dtype=jnp.float4_e2m1fn,
            w2_dtype=jnp.float4_e2m1fn,
            is_quantized=True,
            quant_block_size=512,
            act_fn="silu",
            fp8_direct_write=False,
        )
        return (
            bs.tile_m,
            bs.tile_k1,
            bs.tile_n1,
            bs.tile_k2,
            bs.tile_n2,
            bs.num_w1_bufs,
            bs.num_w2_bufs,
        )

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "256,4096,4096,4096,4096,2,1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "2",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
            },
        ),
    ):
        for m_small in (1024, 4096, 8192):
            bs_small = _query(m_small)
            assert bs_small[0] in (64, 128), (m_small, bs_small)
        for m_mid in (16384, 24576):
            assert _query(m_mid) == (128, 4096, 4096, 4096, 4096, 2, 1)
        for m_large in (32768, 65536, 131072):
            assert _query(m_large) == (256, 4096, 4096, 4096, 4096, 2, 1)


