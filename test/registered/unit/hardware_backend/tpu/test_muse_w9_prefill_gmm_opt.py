# SPDX-License-Identifier: Apache-2.0
"""Unit, Pallas interpret parity, and TPU7x AOT IMEM tests for Wave-9 prefill GMM opts."""

from __future__ import annotations

import ast
import os
import pathlib
from unittest.mock import patch

import numpy as np
import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

_CHECKED_ENV_FLAGS = frozenset({
    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8",
    "SGLANG_TPU_MUSE_KERNEL_GMM_W4A8_K_CHAIN",
    "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU",
    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW",
    "SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS",
    "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS",
    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS",
    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_GMM2_LHS",
})


def test_w9_prefill_gmm_opt_flags_default_off() -> None:
    """Verify Wave-9 prefill GMM flags remain default-off."""
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_W4A8_K_CHAIN.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS.get() == 0.0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_GMM2_LHS.get() == 0.0


def test_no_bare_envfield_bool_eval_in_prefill_gmm_files() -> None:
    """Ensure every EnvField attribute checked here uses explicit `.get()`."""
    repo = pathlib.Path(__file__).resolve().parents[5]
    rel_paths = (
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_v2_gather_scatter.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_fused_rs_nodedup.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/prefill_gmm_opts.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/kernel_b2.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/pipeline.py",
    )
    for rel in rel_paths:
        tree = ast.parse((repo / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _CHECKED_ENV_FLAGS:
                found_get = any(
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "get"
                    and call.func.value is node
                    for call in ast.walk(tree)
                )
                assert found_get, (
                    f"{rel}:{node.lineno} accesses {node.attr} without .get()"
                )


def test_sw8_and_ds1_fused_g2_amax_interpret_parity() -> None:
    """Verify sw8 + ds1 w/ fused g2_amax parity under Pallas InterpretParams."""
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

    rng = np.random.default_rng(20261126)
    # 8 rhs quant blocks (k1=2048, rhs_qbs=256) exercises hoist_rhs_scale=True
    # and 2 N-tiles (512 cols, tile_n1=256) exercises multi-N-tile g2_amax merge.
    m_rows, k1, tile_n1_g, n1_half, k2, n2 = 256, 2048, 512, 512, 512, 256
    lhs_3d = jax.device_put(
        jnp.asarray(
            rng.standard_normal((m_rows, k1 // 128, 128)).astype(np.float32),
            jnp.bfloat16,
        )
    )
    w1_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    s1_bf16 = jnp.asarray(
        np.exp(rng.normal(-2.2, 0.2, size=(k1 // 256, n1_half * 2))).astype(
            np.float32
        ),
        jnp.bfloat16,
    )
    w2_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32)
    )
    s2_bf16 = jnp.asarray(
        np.exp(rng.normal(-2.2, 0.2, size=(k2 // 256, n2))).astype(np.float32),
        jnp.bfloat16,
    )

    w1_tiles = []
    s1_tiles = []
    half_tn1 = tile_n1_g // 2
    for n_id in range(n1_half // half_tn1):
        c0 = n_id * half_tn1
        _g_sl = slice(c0, c0 + half_tn1)
        _u_sl = slice(n1_half + c0, n1_half + c0 + half_tn1)
        w1_tiles.append(jnp.concatenate([w1_u32[:, _g_sl], w1_u32[:, _u_sl]], axis=1))
        s1_tiles.append(jnp.concatenate([s1_bf16[:, _g_sl], s1_bf16[:, _u_sl]], axis=1))
    w1_t0, w1_t1 = w1_tiles
    s1_t0, s1_t1 = s1_tiles

    cfgs1 = GmmConfigs(
        dims=Dimensions(
            size_m=8192,
            size_k=k1,
            size_n=n1_half * 2,
            size_group=1,
            size_lhs_group=1,
            size_lhs_sublane=8,
        ),
        tiles=TileSizes(tile_m=m_rows, tile_k=k1, tile_n=tile_n1_g),
        lhs_cfgs=InputConfigs(
            quant_dtype=jnp.float8_e4m3fn.dtype,
            quant_block_size=k1,
            dtype=jnp.bfloat16,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=jnp.float4_e2m1fn.dtype,
            quant_block_size=256,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k1 // 256,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act="silu",
    )
    cfgs2 = GmmConfigs(
        dims=Dimensions(
            size_m=8192,
            size_k=k2,
            size_n=n2,
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
            quant_block_size=256,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k2 // 256,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act=None,
    )
    eager_interp = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )

    def _run_pair(
        sw_mode: str,
        ds_mode: str,
        flq_rmw: str,
        *,
        skip_mask: bool = True,
    ) -> np.ndarray:
        def _ker(
            x3d_r,
            w1_0_r,
            s1_0_r,
            w1_1_r,
            s1_1_r,
            w2_r,
            s2_r,
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
            m_off[1] = jnp.int32(m_rows if skip_mask else (m_rows - 19))
            meta = MetadataRef(gm_id_to_group_id=m_gid, gm_id_to_m_offset=m_off)
            for n_id, (w_r, s_r) in enumerate(((w1_0_r, s1_0_r), (w1_1_r, s1_1_r))):
                c0 = n_id * half_tn1
                inner_kernel(
                    None,
                    WeightsRef(weight=w_r, scale=s_r, bias=None),
                    g1_r.at[:, pl.ds(c0, half_tn1)],
                    p1_r,
                    acc_r,
                    meta,
                    cfgs=cfgs1,
                    scatter_mode=True,
                    _k_id=0,
                    _num_k=1,
                    _gm_id=0,
                    _n_id=n_id,
                    _num_n=2,
                    fused_g2_amax_slot=1,
                    prequant_lhs_fp8_ref=pq_fp8_r,
                    prequant_lhs_scale_ref=pq_sc_r,
                    skip_row_mask=skip_mask,
                    lhs_3d_vmem_ref=x3d_r,
                )
            inner_kernel(
                None,
                WeightsRef(weight=w2_r, scale=s2_r, bias=None),
                out_r,
                p1_r,
                acc_r,
                meta,
                cfgs=cfgs2,
                scatter_mode=True,
                _k_id=0,
                _num_k=1,
                _gm_id=0,
                _n_id=0,
                _num_n=1,
                fused_g2_amax_slot=1,
                prequant_lhs_fp8_ref=pq_fp8_r.at[:, :k2],
                prequant_lhs_scale_ref=pq_sc_r,
                skip_row_mask=skip_mask,
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
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": sw_mode,
                    "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": ds_mode,
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": flq_rmw,
                },
            ),
        ):
            call = pl.pallas_call(
                _ker,
                out_shape=jax.ShapeDtypeStruct((m_rows, n2), jnp.bfloat16),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 7,
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((m_rows, n1_half), jnp.bfloat16),
                    pltpu.VMEM((m_rows, k1), jnp.float8_e4m3fn),
                    pltpu.VMEM((2, m_rows, 128), jnp.float32),
                    pltpu.VMEM((8, tile_n1_g), jnp.float32),
                    pltpu.VMEM((m_rows, tile_n1_g), jnp.float32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                ],
                interpret=eager_interp,
            )
            return np.asarray(
                call(lhs_3d, w1_t0, s1_t0, w1_t1, s1_t1, w2_u32, s2_bf16),
                dtype=np.float32,
            )

    y_sw1_ds0 = _run_pair("1", "0", "3")
    y_sw8_ds0 = _run_pair("8", "0", "3")
    np.testing.assert_array_equal(y_sw8_ds0, y_sw1_ds0)

    y_sw1_ds1_unfused = _run_pair("1", "1", "0")
    y_sw8_ds1_fused = _run_pair("8", "1", "3")
    assert np.all(np.isfinite(y_sw8_ds1_fused))
    np.testing.assert_array_equal(y_sw8_ds1_fused, y_sw1_ds1_unfused)

    cos = float(
        np.dot(y_sw8_ds1_fused.ravel(), y_sw8_ds0.ravel())
        / (
            np.linalg.norm(y_sw8_ds1_fused.ravel())
            * np.linalg.norm(y_sw8_ds0.ravel())
            + 1e-30
        )
    )
    assert cos > 0.9995, f"Unexpected cosine drift: {cos}"

    y_masked_unfused = _run_pair("1", "1", "0", skip_mask=False)
    y_masked_fused = _run_pair("8", "1", "3", skip_mask=False)
    np.testing.assert_array_equal(y_masked_fused, y_masked_unfused)
    assert np.all(y_masked_fused[-19:] == 0.0)

    # Wave-12: GMM_DIRECT_SILU=2 + GMM_FLQ_ZERO_RMW=3 must fuse g2_amax (0 NaNs).
    y_sw1_ds2_unfused = _run_pair("1", "2", "0")
    y_sw8_ds2_fused = _run_pair("8", "2", "3")
    assert np.all(np.isfinite(y_sw8_ds2_fused))
    np.testing.assert_array_equal(y_sw8_ds2_fused, y_sw1_ds2_unfused)

    y_ds2_masked_unfused = _run_pair("1", "2", "0", skip_mask=False)
    y_ds2_masked_fused = _run_pair("8", "2", "3", skip_mask=False)
    assert np.all(np.isfinite(y_ds2_masked_fused))
    np.testing.assert_array_equal(y_ds2_masked_fused, y_ds2_masked_unfused)
    assert np.all(y_ds2_masked_fused[-19:] == 0.0)


def test_sw8_ds1_tpu7x_aot_compile_and_imem_budget() -> None:
    """Verify TPU7x AOT compile & IMEM size <= sw1_ds0_rb4 on 2x2x1 & 2x2x2."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        gmm_v2_fused_rs,
    )

    base_env = {
        "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
        "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
        "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
        "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "4",
        "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "2",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "256,4096,4096,4096,4096,2,1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
    }

    def _compile_bytes(
        topo_name: str, ep_size: int, e_local: int, sw_mode: str, ds_mode: str
    ) -> tuple[int, str]:
        topo = topologies.get_topology_desc(platform="tpu", topology_name=topo_name)
        mesh = Mesh(np.array(topo.devices), ("model",))
        e_global = ep_size * e_local
        n_slots = e_local + 260
        with patch.dict(
            os.environ,
            {
                **base_env,
                "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": sw_mode,
                "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": ds_mode,
            },
        ):
            jax.clear_caches()

            def _fn(h, hs_sc, w1, w1s, w2, w2s, gs, li, g_st, go, mn, pnw):
                return gmm_v2_fused_rs(
                    h,
                    w1,
                    w2,
                    gs,
                    li,
                    li,
                    w1_scale=w1s,
                    w2_scale=w2s,
                    act_fn="silu",
                    group_offset=go,
                    group_start=g_st,
                    output_size=24576,
                    topk_indices=jnp.zeros((1,), jnp.int32),
                    ep_size=ep_size,
                    ep_axis_name="model",
                    max_num_gm=mn[0],
                    caller_tile_m=256,
                    top_k=8,
                    post_expert_norm_weight=pnw,
                    block_scatter=True,
                    hidden_states_scale=hs_sc,
                )

            in_sp = (
                P(),
                P(),
                P("model"),
                P("model"),
                P("model"),
                P("model"),
                P(),
                P(),
                P(),
                P("model"),
                P(),
                P(),
            )
            sd = jax.ShapeDtypeStruct
            sh_r = NamedSharding(mesh, P())
            sh_m = NamedSharding(mesh, P("model"))
            specs = [
                sd((24576, 32, 128), jnp.float8_e4m3fn, sharding=sh_r),
                sd((24576,), jnp.float32, sharding=sh_r),
                sd((e_global, 4096, 8192), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, 8, 8192), jnp.bfloat16, sharding=sh_m),
                sd((e_global, 4096, 4096), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, 8, 4096), jnp.bfloat16, sharding=sh_m),
                sd((e_global,), jnp.int32, sharding=sh_r),
                sd((196608,), jnp.int32, sharding=sh_r),
                sd((4 + 2 * n_slots,), jnp.int32, sharding=sh_r),
                sd((ep_size,), jnp.int32, sharding=sh_m),
                sd((1,), jnp.int32, sharding=sh_r),
                sd((4096,), jnp.bfloat16, sharding=sh_r),
            ]
            lowered = (
                jax.jit(
                    jax.shard_map(
                        _fn,
                        mesh=mesh,
                        in_specs=in_sp,
                        out_specs=P("model", None, None),
                        check_vma=False,
                    )
                )
                .trace(*specs)
                .lower()
            )
            hlo = lowered.as_text("stablehlo")
            compiled = lowered.compile()
            code_b = int(
                getattr(compiled.memory_analysis(), "generated_code_size_in_bytes", -1)
            )
            return code_b, hlo

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        try:
            code_base, hlo_base = _compile_bytes("tpu7x=2x2x1", 8, 36, "1", "0")
            code_sw8, hlo_sw8 = _compile_bytes("tpu7x=2x2x1", 8, 36, "8", "1")
            assert "-rb4-" in hlo_base and "-sw1-" in hlo_base
            assert "-rb4-" in hlo_sw8 and "-sw8-" in hlo_sw8 and "-ds1-" in hlo_sw8
            assert 0 < code_sw8 <= code_base, (
                f"IMEM regression: sw8+ds1 ({code_sw8} B) > base ({code_base} B)"
            )

            code_ep16, hlo_ep16 = _compile_bytes("tpu7x=2x2x2", 16, 18, "8", "1")
            assert "-rb4-" in hlo_ep16 and "-sw8-" in hlo_ep16 and "-ds1-" in hlo_ep16
            assert 0 < code_ep16 <= 4 * 1024 * 1024
        finally:
            jax.clear_caches()


def test_inter32_closed_form_meta_and_interpret_parity() -> None:
    """Verify inter32 closed-form metadata and Pallas InterpretParams gather/scale/scatter parity."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        _precompute_block_scatter_meta_closed_form,
        _precompute_fused_rs_metadata,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_v2_gather_scatter import (
        MetadataRef,
        dma_gather_gm_start,
        dma_gather_gm_wait,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
        fill_pq_lhs_scales_tree,
    )

    rng = np.random.default_rng(20261127)
    for ep_size, tile_m, my_id, local_start in (
        (8, 256, 3, 128),
        (16, 256, 5, 0),
        (8, 128, 3, 128),
    ):
        size_group = 6
        blk = tile_m // ep_size
        chunk_size = 512
        # Construct counts covering mode 0 (e0: small, e1: >=32 on all but one chip),
        # and mode 1 + mode 2 tail (e2: multiple full blocks + uneven tails,
        # e3: > 1*blk on one chip remaining in tail, e4/e5: additional mode 0 tails).
        counts_gc = np.zeros((size_group, ep_size), dtype=np.int32)
        counts_gc[0] = np.full((ep_size,), 5, dtype=np.int32)
        counts_gc[1] = np.full((ep_size,), blk + 2, dtype=np.int32)
        counts_gc[1, (my_id + 2) % ep_size] = blk - 1
        counts_gc[2] = 3 * blk + (np.arange(ep_size) % 7)
        counts_gc[3] = 2 * blk + 1
        counts_gc[3, (my_id + 1) % ep_size] = 3 * blk + 17
        counts_gc[3, (my_id + 4) % ep_size] = 2 * blk
        counts_gc[4] = np.full((ep_size,), 9, dtype=np.int32)
        counts_gc[5] = np.full((ep_size,), 3 * blk + 3, dtype=np.int32)

        rot_order = [(my_id + 1 + c) % ep_size for c in range(ep_size)]
        per_chip_cursor = np.zeros((ep_size,), dtype=np.int32) + 17
        chip_before = per_chip_cursor.copy()
        idx_parts = []
        if local_start > 0:
            idx_parts.append(np.zeros((local_start,), dtype=np.int32))
        for g in range(size_group):
            for c_tgt in rot_order:
                n_r = int(counts_gc[g, c_tgt])
                if n_r > 0:
                    offs = np.arange(n_r, dtype=np.int32) % chunk_size
                    idx_parts.append(c_tgt * chunk_size + offs)
                    per_chip_cursor[c_tgt] += n_r
        lhs_indices_np = np.concatenate(idx_parts, axis=0)
        sizes_loc = counts_gc.sum(axis=1).astype(np.int32)
        group_sizes = np.zeros((ep_size * size_group,), dtype=np.int32)
        group_sizes[my_id * size_group : (my_id + 1) * size_group] = sizes_loc
        num_slots = size_group + int((sizes_loc.sum() + tile_m - 1) // tile_m) + 4
        group_start_raw = np.concatenate(
            [
                np.array([local_start, int(sizes_loc.sum())], dtype=np.int32),
                chip_before,
                counts_gc.reshape(-1),
            ]
        )

        # With min_kf=2: e1(kf=0), e2(kf=3->inter), e3(kf=2->inter), e4(kf=0), e5(kf=3->inter).
        meta_base = np.asarray(
            _precompute_block_scatter_meta_closed_form(
                jnp.asarray(group_sizes),
                jnp.asarray([my_id * size_group], jnp.int32),
                jnp.asarray(group_start_raw),
                size_group=size_group,
                tile_m=tile_m,
                num_slots=num_slots,
                ep_size=ep_size,
                rotate_rs=True,
                inter32_min_kf=0,
            )
        )
        flat_i32 = np.asarray(
            _precompute_block_scatter_meta_closed_form(
                jnp.asarray(group_sizes),
                jnp.asarray([my_id * size_group], jnp.int32),
                jnp.asarray(group_start_raw),
                size_group=size_group,
                tile_m=tile_m,
                num_slots=num_slots,
                ep_size=ep_size,
                rotate_rs=True,
                inter32_min_kf=2,
            )
        )
        bs_rows = flat_i32[: (num_slots + 1) * (3 * ep_size + 2)].reshape(
            num_slots + 1, 3 * ep_size + 2
        )
        aux_rows = flat_i32[(num_slots + 1) * (3 * ep_size + 2) :].reshape(num_slots, 1 + ep_size)
        modes = aux_rows[:, 0]
        assert np.any(modes == 0) and np.any(modes == 1) and np.any(modes == 2)
        # Semaphore conservation: sent blocks + extra_delta must match total counts per chip.
        has_c = (bs_rows[:num_slots, 2 * ep_size : 3 * ep_size] > 0).astype(np.int32)
        np.testing.assert_array_equal(
            bs_rows[num_slots, :ep_size] + has_c.sum(axis=0), counts_gc.sum(axis=0)
        )
        if local_start > 0:
            shifted_i32 = np.asarray(
                _precompute_block_scatter_meta_closed_form(
                    jnp.asarray(group_sizes),
                    jnp.asarray([my_id * size_group], jnp.int32),
                    jnp.asarray(group_start_raw),
                    size_group=size_group,
                    tile_m=tile_m,
                    num_slots=num_slots,
                    ep_size=ep_size,
                    rotate_rs=True,
                    inter32_min_kf=2,
                    smem_base_offset=local_start,
                )
            )
            shifted_aux = shifted_i32[(num_slots + 1) * (3 * ep_size + 2) :].reshape(
                num_slots, 1 + ep_size
            )
            np.testing.assert_array_equal(shifted_aux[:, 0], aux_rows[:, 0])
            np.testing.assert_array_equal(shifted_aux[:, 1:], aux_rows[:, 1:] - local_start)

        # Verify bit-exact 2-ahead gather + scale pack + block-scatter under eager InterpretParams!
        if ep_size == 8:
            fused_meta = np.asarray(
                _precompute_fused_rs_metadata(
                    jnp.asarray(group_sizes),
                    jnp.asarray([my_id * size_group], jnp.int32),
                    jnp.asarray(group_start_raw),
                    jnp.asarray(lhs_indices_np),
                    size_group=size_group,
                    tile_m=tile_m,
                    sublane=8,
                    chunk_size=chunk_size,
                    num_slots=num_slots,
                )
            )
            num_gm = int(fused_meta[3])
            gm_gid = fused_meta[4 : 4 + num_slots]
            gm_moff = fused_meta[4 + num_slots :]
            n_tok_global = ep_size * chunk_size
            hs_arr = jnp.asarray(
                rng.integers(-120, 120, size=(n_tok_global, 1, 128), dtype=np.int32),
                jnp.float8_e4m3fn,
            )
            row_sc_arr = jnp.asarray(
                rng.uniform(0.1, 2.5, size=(n_tok_global,)).astype(np.float32),
                jnp.float32,
            )
            max_out_rows = int((chip_before + counts_gc.sum(axis=0)).max()) + 32

            def _run_interpret(bs_meta_vec: np.ndarray, *, use_inter32: bool) -> np.ndarray:
                eager_interp = pltpu.InterpretParams(
                    dma_execution_mode="eager",
                    detect_races=True,
                    uninitialized_memory="nan",
                )

                def _ker(
                    lhs_i_r,
                    bs_r,
                    gid_r,
                    moff_r,
                    sc_r,
                    hs_r,
                    out_r,
                    g_lhs_2x,
                    pq_sc_2x,
                    g_sem,
                    tile_smem_r,
                ):
                    meta_ref = MetadataRef(gm_id_to_group_id=gid_r, gm_id_to_m_offset=moff_r)
                    i32_g_kw = (
                        {
                            "bs_meta_ref": bs_r,
                            "ep_size": ep_size,
                            "my_id": my_id,
                            "tile_idx_smem_ref": tile_smem_r,
                        }
                        if use_inter32
                        else {}
                    )
                    out_r[:, :] = jnp.zeros_like(out_r)

                    def _pro(gi, _u):
                        dma_gather_gm_start(
                            hs_r,
                            g_lhs_2x.at[gi],
                            lhs_i_r,
                            g_sem.at[gi],
                            gi,
                            meta_ref,
                            divisor=1,
                            packed_u16=False,
                            **i32_g_kw,
                        )
                        return _u

                    lax.fori_loop(0, min(num_gm, 2), _pro, jnp.int32(0))

                    def _body(gm_id, _u):
                        sem_id = gm_id % 2
                        m_st = moff_r[gm_id]
                        m_en = moff_r[gm_id + 1]
                        nv = m_en - m_st
                        dma_gather_gm_wait(g_lhs_2x.at[sem_id], g_sem.at[sem_id], gm_id, meta_ref)
                        i32_s_kw = (
                            {
                                "bs_meta_ref": bs_r,
                                "gm_id": gm_id,
                                "n_slots_gm": num_slots,
                                "ep_size": ep_size,
                                "tile_idx_smem_ref": tile_smem_r,
                            }
                            if use_inter32
                            else {}
                        )
                        fill_pq_lhs_scales_tree(
                            pq_sc_2x,
                            sc_r,
                            lhs_i_r,
                            m_st=m_st,
                            m_en=m_en,
                            num_valid=nv,
                            num_lanes=128,
                            packed_u16_lhs=False,
                            gather_divisor=1,
                            scale_b_id=0,
                            unroll_m_rows=tile_m,
                            **i32_s_kw,
                        )
                        # Consume gathered LHS[sem_id] before launching 2-ahead DMA into sem_id.
                        tile_f32 = (
                            g_lhs_2x[sem_id, :, 0, :].astype(jnp.float32)
                            * pq_sc_2x[0, :, :1]
                        )

                        @pl.when(gm_id + 2 < num_gm)
                        def _():
                            dma_gather_gm_start(
                                hs_r,
                                g_lhs_2x.at[sem_id],
                                lhs_i_r,
                                g_sem.at[sem_id],
                                gm_id + 2,
                                meta_ref,
                                divisor=1,
                                packed_u16=False,
                                **i32_g_kw,
                            )

                        bs_base = gm_id * (3 * ep_size + 2)

                        def _sc_row(r_i, _u2):
                            c_tgt = jnp.mod(my_id + 1 + r_i, ep_size)
                            s0 = bs_r[bs_base + c_tgt]
                            d0 = bs_r[bs_base + ep_size + c_tgt]
                            ln = bs_r[bs_base + 2 * ep_size + c_tgt]
                            global_off = c_tgt * jnp.int32(max_out_rows) + d0

                            def _cp_r(j, _u3):
                                out_r[global_off + j, :] = tile_f32[s0 + j, :]
                                return _u3

                            lax.fori_loop(0, ln, _cp_r, jnp.int32(0))
                            return _u2

                        lax.fori_loop(0, ep_size, _sc_row, jnp.int32(0))
                        return _u

                    lax.fori_loop(0, num_gm, _body, jnp.int32(0))

                with (
                    patch(
                        "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                        return_value="TPU v7",
                    ),
                    tpu_info_for_cpu_export(),
                ):
                    res = pl.pallas_call(
                        _ker,
                        out_shape=jax.ShapeDtypeStruct(
                            (ep_size * max_out_rows, 128), jnp.float32
                        ),
                        grid_spec=pltpu.PrefetchScalarGridSpec(
                            num_scalar_prefetch=5,
                            grid=(1,),
                            in_specs=[
                                pl.BlockSpec(memory_space=pltpu.HBM),
                            ],
                            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                            scratch_shapes=[
                                pltpu.VMEM((2, tile_m, 1, 128), jnp.float8_e4m3fn),
                                pltpu.VMEM((1, tile_m, 128), jnp.float32),
                                pltpu.SemaphoreType.DMA((2,)),
                                pltpu.SMEM((4, tile_m), jnp.int32),
                            ],
                        ),
                        interpret=eager_interp,
                    )(
                        jnp.asarray(lhs_indices_np),
                        jnp.asarray(bs_meta_vec.reshape(-1)),
                        jnp.asarray(gm_gid),
                        jnp.asarray(gm_moff),
                        row_sc_arr,
                        hs_arr,
                    )
                return np.asarray(res, dtype=np.float32)

            ref_out = _run_interpret(meta_base.reshape(-1), use_inter32=False)
            got_out = _run_interpret(flat_i32.reshape(-1), use_inter32=True)
            assert np.all(np.isfinite(got_out))
            np.testing.assert_array_equal(got_out, ref_out)


def test_inter32_tpu7x_aot_compile_and_imem_budget() -> None:
    """Verify inter32 flag is opt-in, tags HLO cleanly, and meets TPU7x IMEM caps."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        gmm_v2_fused_rs,
    )

    base_env = {
        "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "8",
        "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": "1",
        "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
        "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
        "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "4",
        "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "2",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "256,4096,4096,4096,4096,2,1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
        "SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS": "-1",
    }

    def _compile_bytes(
        topo_name: str, ep_size: int, e_local: int, inter32_mode: str
    ) -> tuple[int, str]:
        topo = topologies.get_topology_desc(platform="tpu", topology_name=topo_name)
        mesh = Mesh(np.array(topo.devices), ("model",))
        e_global = ep_size * e_local
        n_slots = e_local + 260
        with patch.dict(
            os.environ,
            {**base_env, "SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS": inter32_mode},
        ):
            jax.clear_caches()

            def _fn(h, hs_sc, w1, w1s, w2, w2s, gs, li, g_st, go, mn, pnw):
                return gmm_v2_fused_rs(
                    h,
                    w1,
                    w2,
                    gs,
                    li,
                    li,
                    w1_scale=w1s,
                    w2_scale=w2s,
                    act_fn="silu",
                    group_offset=go,
                    group_start=g_st,
                    output_size=24576,
                    topk_indices=jnp.zeros((1,), jnp.int32),
                    ep_size=ep_size,
                    ep_axis_name="model",
                    max_num_gm=mn[0],
                    caller_tile_m=256,
                    top_k=8,
                    post_expert_norm_weight=pnw,
                    block_scatter=True,
                    hidden_states_scale=hs_sc,
                )

            in_sp = (
                P(),
                P(),
                P("model"),
                P("model"),
                P("model"),
                P("model"),
                P(),
                P(),
                P(),
                P("model"),
                P(),
                P(),
            )
            sd = jax.ShapeDtypeStruct
            sh_r = NamedSharding(mesh, P())
            sh_m = NamedSharding(mesh, P("model"))
            specs = [
                sd((24576, 32, 128), jnp.float8_e4m3fn, sharding=sh_r),
                sd((24576,), jnp.float32, sharding=sh_r),
                sd((e_global, 4096, 8192), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, 8, 8192), jnp.bfloat16, sharding=sh_m),
                sd((e_global, 4096, 4096), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, 8, 4096), jnp.bfloat16, sharding=sh_m),
                sd((e_global,), jnp.int32, sharding=sh_r),
                sd((196608,), jnp.int32, sharding=sh_r),
                sd((4 + 2 * n_slots,), jnp.int32, sharding=sh_r),
                sd((ep_size,), jnp.int32, sharding=sh_m),
                sd((1,), jnp.int32, sharding=sh_r),
                sd((4096,), jnp.bfloat16, sharding=sh_r),
            ]
            lowered = (
                jax.jit(
                    jax.shard_map(
                        _fn,
                        mesh=mesh,
                        in_specs=in_sp,
                        out_specs=P("model", None, None),
                        check_vma=False,
                    )
                )
                .trace(*specs)
                .lower()
            )
            hlo = lowered.as_text("stablehlo")
            compiled = lowered.compile()
            code_b = int(
                getattr(compiled.memory_analysis(), "generated_code_size_in_bytes", -1)
            )
            return code_b, hlo

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        try:
            code_off, hlo_off = _compile_bytes("tpu7x=2x2x1", 8, 36, "0")
            code_i32, hlo_i32 = _compile_bytes("tpu7x=2x2x1", 8, 36, "12")
            assert "-i32_" not in hlo_off
            assert "-i32_12" in hlo_i32
            assert 0 < code_off <= 3_385_344
            assert 0 < code_i32 <= 3_385_344, (
                f"IMEM regression: inter32 ({code_i32} B) > 3,385,344 B cap"
            )

            code_ep16, hlo_ep16 = _compile_bytes("tpu7x=2x2x2", 16, 18, "12")
            assert "-i32_12" in hlo_ep16
            assert 0 < code_ep16 <= 4 * 1024 * 1024
        finally:
            jax.clear_caches()


def test_resolve_static_fp8_lhs_scale() -> None:
    """Verify static FP8 LHS and GMM2 scale resolution and range bounds."""
    from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
        resolve_static_fp8_gmm2_scale,
        resolve_static_fp8_lhs_scale,
    )

    assert resolve_static_fp8_lhs_scale(None, None) == 0.0
    assert resolve_static_fp8_lhs_scale(0, 0.0) == 0.0
    assert resolve_static_fp8_lhs_scale(5, 0.0) == 0.0
    assert resolve_static_fp8_lhs_scale(6, 0.0) == 8.0
    assert resolve_static_fp8_lhs_scale(6, None) == 8.0
    assert resolve_static_fp8_lhs_scale(6, 4.0) == 4.0
    assert resolve_static_fp8_lhs_scale(0, 12.0) == 12.0
    with pytest.raises(ValueError, match="PREFILL_STATIC_FP8_LHS"):
        resolve_static_fp8_lhs_scale(0, -1.0)
    with pytest.raises(ValueError, match="PREFILL_STATIC_FP8_LHS"):
        resolve_static_fp8_lhs_scale(0, 33.0)
    with pytest.raises(ValueError, match="PREFILL_STATIC_FP8_LHS"):
        resolve_static_fp8_lhs_scale(0, float("nan"))

    assert resolve_static_fp8_gmm2_scale(None) == 0.0
    assert resolve_static_fp8_gmm2_scale(0.0) == 0.0
    assert resolve_static_fp8_gmm2_scale(2.0) == 2.0
    assert resolve_static_fp8_gmm2_scale(4) == 4.0
    with pytest.raises(ValueError, match="PREFILL_STATIC_FP8_GMM2_LHS"):
        resolve_static_fp8_gmm2_scale(-0.5)
    with pytest.raises(ValueError, match="PREFILL_STATIC_FP8_GMM2_LHS"):
        resolve_static_fp8_gmm2_scale(32.5)
    with pytest.raises(ValueError, match="PREFILL_STATIC_FP8_GMM2_LHS"):
        resolve_static_fp8_gmm2_scale(float("nan"))


def test_kernel_b2_static_fp8_interpret_and_dbl_stream() -> None:
    """Verify kernel_b2 static FP8 1-pass quant parity for both dbl_stream False/True and row_range."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels import kernel_b2

    real_pc = pl.pallas_call

    def _eager_interp(*args, **kwargs):
        kwargs["interpret"] = pltpu.InterpretParams(
            dma_execution_mode="eager",
            detect_races=True,
            uninitialized_memory="nan",
        )
        return real_pc(*args, **kwargs)

    rng = np.random.default_rng(20261128)
    t_loc, h_dim, he_dim, n_exp = 384, 8192, 4096, 256
    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        patch.object(pl, "pallas_call", _eager_interp),
        patch.object(tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST, "get", return_value=True),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        attn = jnp.array(
            (rng.standard_normal((t_loc, h_dim)) * 0.25).astype(np.float32)
        ).astype(jnp.bfloat16)
        stream = jnp.array(
            (rng.standard_normal((t_loc, h_dim)) * 0.30).astype(np.float32)
        )
        v8k = jnp.array(rng.uniform(0.5, 1.5, (h_dim,)).astype(np.float32))
        wr = jnp.array((rng.standard_normal((h_dim, n_exp)) * 0.05).astype(np.float32))
        eb = jnp.zeros((n_exp,), dtype=jnp.float32)
        wp = jnp.array(
            (rng.standard_normal((h_dim, he_dim)) * 0.20).astype(np.float32)
        ).astype(jnp.float8_e4m3fn)
        v4k_f = jnp.array(rng.uniform(0.5, 1.5, (he_dim,)).astype(np.float32))
        # Match Muse Spark checkpoint pre_expert_norm_w dynamic range (-0.9 .. +3.6 -> +1 = 0.1 .. 4.6).
        pe_w = jnp.array(rng.uniform(-0.8, 3.6, (he_dim,)).astype(np.float32)).astype(
            jnp.bfloat16
        )

        with pytest.raises(ValueError, match="static_fp8_scale"):
            kernel_b2.pallas_fused_b1_b2_wpool_quant(
                attn,
                stream,
                v8k,
                v8k,
                v8k,
                wr,
                eb,
                wp,
                v4k_f,
                pe_w,
                emit_fp8_gmm=True,
                static_fp8_scale=64.0,
            )

        with patch.object(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_B2_DBL_STREAM, "get", return_value=True
        ):
            dyn_fp8, dyn_sc, dyn_sm, dyn_id, dyn_tw = (
                kernel_b2.pallas_fused_b1_b2_wpool_quant(
                    attn,
                    stream,
                    v8k,
                    v8k,
                    v8k,
                    wr,
                    eb,
                    wp,
                    v4k_f,
                    pe_w,
                    router_mode="bf16_hilo",
                    fp8_wpool=True,
                    emit_fp8_gmm=True,
                )
            )

        dyn_deq = np.asarray(
            dyn_fp8.reshape(t_loc, he_dim).astype(jnp.float32) * dyn_sc[:, None]
        )
        static_fp8_outputs = []
        for dbl_flag in (False, True):
            with patch.object(
                tpu_envs.SGLANG_TPU_MUSE_KERNEL_B2_DBL_STREAM,
                "get",
                return_value=dbl_flag,
            ):
                st_fp8, st_sc, st_sm, st_id, st_tw = (
                    kernel_b2.pallas_fused_b1_b2_wpool_quant(
                        attn,
                        stream,
                        v8k,
                        v8k,
                        v8k,
                        wr,
                        eb,
                        wp,
                        v4k_f,
                        pe_w,
                        router_mode="bf16_hilo",
                        fp8_wpool=True,
                        emit_fp8_gmm=True,
                        static_fp8_scale=8.0,
                    )
                )
            assert st_sc is None
            np.testing.assert_array_equal(np.asarray(st_sm), np.asarray(dyn_sm))
            np.testing.assert_array_equal(np.asarray(st_id), np.asarray(dyn_id))
            np.testing.assert_array_equal(
                np.asarray(st_tw.astype(jnp.float32)),
                np.asarray(dyn_tw.astype(jnp.float32)),
            )
            st_deq = np.asarray(st_fp8.reshape(t_loc, he_dim).astype(jnp.float32)) / 8.0
            assert np.all(np.isfinite(st_deq))
            cos = float(
                np.dot(st_deq.ravel(), dyn_deq.ravel())
                / (np.linalg.norm(st_deq.ravel()) * np.linalg.norm(dyn_deq.ravel()))
            )
            assert cos >= 0.999, f"dbl_stream={dbl_flag} static_fp8 cos={cos} < 0.999"
            static_fp8_outputs.append(np.asarray(st_fp8.astype(jnp.float32)))
        np.testing.assert_array_equal(static_fp8_outputs[0], static_fp8_outputs[1])

        # row_range chaining parity with B2_DBL_STREAM=True + static_fp8_scale=8.0.
        with patch.object(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_B2_DBL_STREAM, "get", return_value=True
        ):
            r0 = kernel_b2.pallas_fused_b1_b2_wpool_quant(
                attn,
                stream,
                v8k,
                v8k,
                v8k,
                wr,
                eb,
                wp,
                v4k_f,
                pe_w,
                router_mode="bf16_hilo",
                fp8_wpool=True,
                row_range=(0, 128),
                full_stream_mid=True,
                emit_fp8_gmm=True,
                static_fp8_scale=8.0,
            )
            r1 = kernel_b2.pallas_fused_b1_b2_wpool_quant(
                attn,
                stream,
                v8k,
                v8k,
                v8k,
                wr,
                eb,
                wp,
                v4k_f,
                pe_w,
                router_mode="bf16_hilo",
                fp8_wpool=True,
                row_range=(128, t_loc),
                stream_mid_prev=r0[2],
                emit_fp8_gmm=True,
                static_fp8_scale=8.0,
            )
            assert r0[1] is None and r1[1] is None
            np.testing.assert_array_equal(
                np.asarray(jnp.concatenate([r0[0], r1[0]], axis=0).astype(jnp.float32)),
                np.asarray(st_fp8.astype(jnp.float32)),
            )
            np.testing.assert_array_equal(np.asarray(r1[2]), np.asarray(st_sm))

        # Omitting static_fp8_scale (`None`) auto-resolves from env (`pql=6` -> 8.0),
        # so callers like B2_PIPE_CHUNKS compose cleanly without extra args.
        with (
            patch.object(
                tpu_envs.SGLANG_TPU_MUSE_KERNEL_B2_DBL_STREAM, "get", return_value=True
            ),
            patch.object(
                tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8,
                "get",
                return_value=6,
            ),
        ):
            auto_res = kernel_b2.pallas_fused_b1_b2_wpool_quant(
                attn,
                stream,
                v8k,
                v8k,
                v8k,
                wr,
                eb,
                wp,
                v4k_f,
                pe_w,
                router_mode="bf16_hilo",
                fp8_wpool=True,
                emit_fp8_gmm=True,
            )
            assert auto_res[1] is None
            np.testing.assert_array_equal(
                np.asarray(auto_res[0].astype(jnp.float32)),
                np.asarray(st_fp8.astype(jnp.float32)),
            )


def test_gmm_static_fp8_lhs_skip_scale_and_gs_fold_interpret() -> None:
    """Verify skip_lhs_scale_mul leaves scale slot 0 unread (NaN-safe) and 1/S w1_gs matches."""
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
    )

    rng = np.random.default_rng(20261129)
    m_rows, k1, n1_half, n2 = 256, 512, 256, 256
    tile_n1_g = 256
    k2 = n1_half

    lhs_fp8 = jnp.asarray(
        (rng.standard_normal((m_rows, k1)) * 12.0).clip(-440, 440), jnp.float8_e4m3fn
    )
    w1_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    s1_bf16 = jnp.asarray(
        rng.uniform(0.05, 0.25, size=(k1 // 256, n1_half * 2)).astype(np.float32),
        jnp.bfloat16,
    )
    w2_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32)
    )
    s2_bf16 = jnp.asarray(
        rng.uniform(0.05, 0.25, size=(k2 // 256, n2)).astype(np.float32),
        jnp.bfloat16,
    )
    static_s = 8.0
    inv_s_arr = jnp.asarray([1.0 / static_s], dtype=jnp.float32)
    nan_arr = jnp.full((2, m_rows, 128), np.nan, dtype=np.float32)

    cfgs1 = GmmConfigs(
        dims=Dimensions(8192, k1, n1_half * 2, 1, 1, 8),
        tiles=TileSizes(m_rows, k1, n1_half),
        lhs_cfgs=InputConfigs(
            quant_dtype=jnp.float8_e4m3fn.dtype,
            quant_block_size=k1,
            dtype=jnp.bfloat16,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=jnp.float4_e2m1fn.dtype,
            quant_block_size=256,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k1 // 256,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act="silu",
    )
    cfgs2 = GmmConfigs(
        dims=Dimensions(8192, k2, n2, 1, 1, 8),
        tiles=TileSizes(m_rows, k2, n2),
        lhs_cfgs=InputConfigs(
            quant_dtype=jnp.float8_e4m3fn.dtype,
            quant_block_size=k2,
            dtype=jnp.bfloat16,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=jnp.float4_e2m1fn.dtype,
            quant_block_size=256,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k2 // 256,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act=None,
    )
    eager_interp = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )

    def _run_case(
        *,
        sw_mode: str,
        ds_mode: str,
        fa_mode: str,
        static_mode: bool,
    ) -> np.ndarray:
        def _ker(
            x_r,
            w1_r,
            s1_r,
            w2_r,
            s2_r,
            gs_r,
            sc_init_r,
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
            m_off[1] = jnp.int32(m_rows)
            meta = MetadataRef(gm_id_to_group_id=m_gid, gm_id_to_m_offset=m_off)
            pq_fp8_r[:, :] = x_r[:, :]
            pq_sc_r[:, :, :] = sc_init_r[:, :, :]
            _g2_slot = 1
            inner_kernel(
                None,
                WeightsRef(weight=w1_r, scale=s1_r, bias=None),
                g1_r,
                p1_r,
                acc_r,
                meta,
                cfgs=cfgs1,
                gs_gate_ref=gs_r if static_mode else None,
                gs_up_ref=gs_r if static_mode else None,
                scatter_mode=True,
                _k_id=0,
                _num_k=1,
                _gm_id=0,
                _n_id=0,
                _num_n=1,
                fused_g2_amax_slot=_g2_slot,
                prequant_lhs_fp8_ref=pq_fp8_r,
                prequant_lhs_scale_ref=pq_sc_r,
                skip_row_mask=True,
                prequant_lhs_ready=True,
                skip_lhs_scale_mul=static_mode,
            )
            inner_kernel(
                None,
                WeightsRef(weight=w2_r, scale=s2_r, bias=None),
                out_r,
                p1_r,
                acc_r,
                meta,
                cfgs=cfgs2,
                scatter_mode=True,
                _k_id=0,
                _num_k=1,
                _gm_id=0,
                _n_id=0,
                _num_n=1,
                fused_g2_amax_slot=_g2_slot,
                prequant_lhs_fp8_ref=pq_fp8_r.at[:, :k2],
                prequant_lhs_scale_ref=pq_sc_r,
                skip_row_mask=True,
                lhs_2d_vmem_ref=g1_r,
            )

        sc_init = (
            nan_arr
            if static_mode
            else jnp.full((2, m_rows, 128), 1.0 / static_s, dtype=jnp.float32)
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
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": sw_mode,
                    "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": ds_mode,
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "512",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": fa_mode,
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                },
            ),
        ):
            call = pl.pallas_call(
                _ker,
                out_shape=jax.ShapeDtypeStruct((m_rows, n2), jnp.bfloat16),
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.SMEM),
                    pl.BlockSpec(memory_space=pltpu.VMEM),
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((m_rows, n1_half), jnp.bfloat16),
                    pltpu.VMEM((m_rows, k1), jnp.float8_e4m3fn),
                    pltpu.VMEM((2, m_rows, 128), jnp.float32),
                    pltpu.VMEM((8, tile_n1_g), jnp.float32),
                    pltpu.VMEM((m_rows, tile_n1_g), jnp.float32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                ],
                interpret=eager_interp,
            )
            return np.asarray(
                call(lhs_fp8, w1_u32, s1_bf16, w2_u32, s2_bf16, inv_s_arr, sc_init),
                dtype=np.float32,
            )

    for sw_m, ds_m, fa_m in (("1", "0", "1"), ("8", "0", "1"), ("8", "1", "1"), ("8", "0", "3")):
        y_dyn = _run_case(sw_mode=sw_m, ds_mode=ds_m, fa_mode=fa_m, static_mode=False)
        y_st = _run_case(sw_mode=sw_m, ds_mode=ds_m, fa_mode=fa_m, static_mode=True)
        assert np.all(np.isfinite(y_st)), f"NaN/Inf in static mode sw={sw_m} ds={ds_m} fa={fa_m}"
        cos = float(
            np.dot(y_st.ravel(), y_dyn.ravel())
            / (np.linalg.norm(y_st.ravel()) * np.linalg.norm(y_dyn.ravel()))
        )
        assert cos >= 0.9999, f"static fold mismatch sw={sw_m} ds={ds_m} fa={fa_m}: cos={cos}"

    # Verify fill_pq_lhs_scales_tree supports inter32 tile_m=128 (EP=8, blk_m=16) cleanly.
    n_tok, ep_size = 64, 8
    scales_in = jnp.asarray(rng.uniform(0.2, 1.5, size=(n_tok,)), dtype=jnp.float32)
    lhs_idx = jnp.arange(n_tok, dtype=jnp.int32)
    n_slots = 1
    bs_meta = np.zeros(((n_slots + 1) * (3 * ep_size + 2) + n_slots * (1 + ep_size),), dtype=np.int32)
    aux_off = (n_slots + 1) * (3 * ep_size + 2)
    bs_meta[aux_off] = 0
    for qb in range(ep_size):
        bs_meta[aux_off + 1 + qb] = min(qb * 16, 32)
    bs_meta_j = jnp.asarray(bs_meta, dtype=jnp.int32)

    def _fill_k(sc_r, li_r, bm_r, out_sc_r, sm_idx_r):
        out_sc_r[...] = jnp.zeros_like(out_sc_r)
        fill_pq_lhs_scales_tree(
            out_sc_r,
            sc_r,
            li_r,
            m_st=jnp.int32(0),
            m_en=jnp.int32(48),
            num_valid=jnp.int32(48),
            num_lanes=128,
            packed_u16_lhs=False,
            gather_divisor=1,
            scale_b_id=0,
            bs_meta_ref=bm_r,
            gm_id=jnp.int32(0),
            n_slots_gm=n_slots,
            ep_size=ep_size,
            tile_idx_smem_ref=sm_idx_r,
        )

    out_sc = pl.pallas_call(
        _fill_k,
        out_shape=jax.ShapeDtypeStruct((1, 128, 128), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=[],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[pltpu.SMEM((4, 128), jnp.int32)],
        ),
        interpret=eager_interp,
    )(scales_in, lhs_idx, bs_meta_j)
    out_sc_np = np.asarray(out_sc[0, :48, 0], dtype=np.float32)
    expected = np.concatenate([
        np.asarray(scales_in[:16]),
        np.asarray(scales_in[16:32]),
        np.asarray(scales_in[32:48]),
    ])
    np.testing.assert_array_equal(out_sc_np, expected)


def test_pql6_and_inter32_tm128_tpu7x_aot_imem_budget() -> None:
    """Verify pql6 saves TPU7x IMEM and Proposal #3 enables inter32 at M=32768 tile_m=128."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        gmm_v2_fused_rs,
    )

    base_env = {
        "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "8",
        "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": "1",
        "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
        "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
        "SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS": "12",
        "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "2",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "256,4096,4096,4096,4096,2,1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
        "SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS": "-1",
    }

    def _compile(
        topo_name: str,
        ep_size: int,
        e_local: int,
        num_tok: int,
        caller_tm: int,
        pq_mode: str,
        static_lhs_s: str,
        with_ext_sc: bool,
        *,
        prefill_tiles: str = base_env["SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES"],
        fa_mode: str = "1",
    ) -> tuple[int, str]:
        topo = topologies.get_topology_desc(platform="tpu", topology_name=topo_name)
        mesh = Mesh(np.array(topo.devices), ("model",))
        e_global = ep_size * e_local
        n_slots = e_local + (num_tok * 8 + caller_tm - 1) // caller_tm
        env_map = {
            **base_env,
            "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": fa_mode,
            "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": pq_mode,
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS": static_lhs_s,
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": prefill_tiles,
        }
        with patch.dict(os.environ, env_map):
            jax.clear_caches()

            def _fn(h, hs_sc, w1, w1s, w2, w2s, gs, li, g_st, go, mn, pnw):
                return gmm_v2_fused_rs(
                    h,
                    w1,
                    w2,
                    gs,
                    li,
                    li,
                    w1_scale=w1s,
                    w2_scale=w2s,
                    act_fn="silu",
                    group_offset=go,
                    group_start=g_st,
                    output_size=num_tok,
                    topk_indices=jnp.zeros((1,), jnp.int32),
                    ep_size=ep_size,
                    ep_axis_name="model",
                    max_num_gm=mn[0],
                    caller_tile_m=caller_tm,
                    top_k=8,
                    post_expert_norm_weight=pnw,
                    block_scatter=True,
                    hidden_states_scale=(hs_sc if with_ext_sc else None),
                )

            in_sp = (
                P(),
                P(),
                P("model"),
                P("model"),
                P("model"),
                P("model"),
                P(),
                P(),
                P(),
                P("model"),
                P(),
                P(),
            )
            sd = jax.ShapeDtypeStruct
            sh_r = NamedSharding(mesh, P())
            sh_m = NamedSharding(mesh, P("model"))
            specs = [
                sd((num_tok, 32, 128), jnp.float8_e4m3fn, sharding=sh_r),
                sd((num_tok,), jnp.float32, sharding=sh_r),
                sd((e_global, 4096, 8192), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, 8, 8192), jnp.bfloat16, sharding=sh_m),
                sd((e_global, 4096, 4096), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, 8, 4096), jnp.bfloat16, sharding=sh_m),
                sd((e_global,), jnp.int32, sharding=sh_r),
                sd((num_tok * 8,), jnp.int32, sharding=sh_r),
                sd((4 + 2 * n_slots,), jnp.int32, sharding=sh_r),
                sd((ep_size,), jnp.int32, sharding=sh_m),
                sd((1,), jnp.int32, sharding=sh_r),
                sd((4096,), jnp.bfloat16, sharding=sh_r),
            ]
            lowered = (
                jax.jit(
                    jax.shard_map(
                        _fn,
                        mesh=mesh,
                        in_specs=in_sp,
                        out_specs=P("model", None, None),
                        check_vma=False,
                    )
                )
                .trace(*specs)
                .lower()
            )
            hlo = lowered.as_text("stablehlo")
            compiled = lowered.compile()
            code_b = int(
                getattr(compiled.memory_analysis(), "generated_code_size_in_bytes", -1)
            )
            return code_b, hlo

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        try:
            code_pql4, hlo_pql4 = _compile("tpu7x=2x2x1", 8, 36, 24576, 256, "4", "0", True)
            code_pql6, hlo_pql6 = _compile("tpu7x=2x2x1", 8, 36, 24576, 256, "6", "0", False)
            assert "-pql4" in hlo_pql4 and "-i32_12" in hlo_pql4
            assert "-pql6-slh8" in hlo_pql6 and "-i32_12" in hlo_pql6
            assert 0 < code_pql6 < code_pql4 <= 3_385_344, (
                f"Expected pql6 ({code_pql6} B) < pql4 ({code_pql4} B) <= 3,385,344 B"
            )

            # Proposal #3: M=32768 (T=4096, tile_m=128, EP=8) activates inter32=12 within cap.
            code_tm128, hlo_tm128 = _compile(
                "tpu7x=2x2x1",
                8,
                36,
                4096,
                128,
                "6",
                "8.0",
                False,
                prefill_tiles="",
            )
            assert "-i32_12" in hlo_tm128 and "-pql6-slh8" in hlo_tm128
            assert 0 < code_tm128 <= 3_385_344

            code_ep16, hlo_ep16 = _compile(
                "tpu7x=2x2x2", 16, 28, 24576, 256, "6", "0", False, fa_mode="3"
            )
            assert "-pql6-slh8" in hlo_ep16 and "-i32_12" in hlo_ep16
            assert "-fa3" in hlo_ep16 and "-E_28-" in hlo_ep16
            assert 0 < code_ep16 <= 3_385_344

            code_fa3, hlo_fa3 = _compile(
                "tpu7x=2x2x1", 8, 36, 24576, 256, "6", "0", False, fa_mode="3"
            )
            assert "-fa3" in hlo_fa3 and "-pql6-slh8" in hlo_fa3
            assert 0 < code_fa3 <= 3_385_344
        finally:
            jax.clear_caches()


def test_prefill_static_fp8_gmm2_lhs_interpret_and_aot() -> None:
    """Verify opt-in PREFILL_STATIC_FP8_GMM2_LHS InterpretParams parity, clipping safety, and TPU7x AOT."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        gmm_v2_fused_rs,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_v2_gather_scatter import (
        Dimensions,
        GmmConfigs,
        InputConfigs,
        MetadataRef,
        TileSizes,
        WeightsRef,
        inner_kernel,
    )

    rng = np.random.default_rng(20261129)
    m_rows, k1, tile_n1_g, n1_half, k2, n2 = 256, 2048, 512, 512, 512, 256
    w1_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    s1_bf16 = jnp.asarray(
        np.exp(rng.normal(-2.2, 0.2, size=(k1 // 256, n1_half * 2))).astype(
            np.float32
        ),
        jnp.bfloat16,
    )
    w2_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32)
    )
    s2_bf16 = jnp.asarray(
        np.exp(rng.normal(-2.2, 0.2, size=(k2 // 256, n2))).astype(np.float32),
        jnp.bfloat16,
    )
    half_tn1 = tile_n1_g // 2
    w1_tiles, s1_tiles = [], []
    for n_id in range(n1_half // half_tn1):
        c0 = n_id * half_tn1
        _g_sl = slice(c0, c0 + half_tn1)
        _u_sl = slice(n1_half + c0, n1_half + c0 + half_tn1)
        w1_tiles.append(jnp.concatenate([w1_u32[:, _g_sl], w1_u32[:, _u_sl]], axis=1))
        s1_tiles.append(jnp.concatenate([s1_bf16[:, _g_sl], s1_bf16[:, _u_sl]], axis=1))
    w1_t0, w1_t1 = w1_tiles
    s1_t0, s1_t1 = s1_tiles

    cfgs1 = GmmConfigs(
        dims=Dimensions(
            size_m=8192,
            size_k=k1,
            size_n=n1_half * 2,
            size_group=1,
            size_lhs_group=1,
            size_lhs_sublane=8,
        ),
        tiles=TileSizes(tile_m=m_rows, tile_k=k1, tile_n=tile_n1_g),
        lhs_cfgs=InputConfigs(
            quant_dtype=jnp.float8_e4m3fn.dtype,
            quant_block_size=k1,
            dtype=jnp.bfloat16,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=jnp.float4_e2m1fn.dtype,
            quant_block_size=256,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k1 // 256,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act="silu",
    )
    cfgs2 = GmmConfigs(
        dims=Dimensions(
            size_m=8192,
            size_k=k2,
            size_n=n2,
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
            quant_block_size=256,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k2 // 256,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act=None,
    )
    eager_interp = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )

    def _run_g2_case(
        lhs_scale_f: float,
        *,
        static_g2_s: float,
        ds_mode: str = "1",
        skip_mask: bool = True,
    ) -> np.ndarray:
        lhs_3d = jax.device_put(
            jnp.asarray(
                rng.standard_normal((m_rows, k1 // 128, 128)).astype(np.float32)
                * lhs_scale_f,
                jnp.bfloat16,
            )
        )
        g1_dtype = jnp.float8_e4m3fn if static_g2_s > 0.0 else jnp.bfloat16
        gs_gate = jnp.asarray([1.0], dtype=jnp.float32)
        gs_up = jnp.asarray(
            [static_g2_s if static_g2_s > 0.0 else 1.0], dtype=jnp.float32
        )
        gs_w2 = jnp.asarray(
            [1.0 / static_g2_s if static_g2_s > 0.0 else 1.0], dtype=jnp.float32
        )

        def _ker(
            x3d_r,
            w1_0_r,
            s1_0_r,
            w1_1_r,
            s1_1_r,
            w2_r,
            s2_r,
            gs_g_r,
            gs_u_r,
            gs_w2_r,
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
            m_off[1] = jnp.int32(m_rows if skip_mask else (m_rows - 19))
            meta = MetadataRef(gm_id_to_group_id=m_gid, gm_id_to_m_offset=m_off)
            for n_id, (w_r, s_r) in enumerate(((w1_0_r, s1_0_r), (w1_1_r, s1_1_r))):
                c0 = n_id * half_tn1
                inner_kernel(
                    None,
                    WeightsRef(weight=w_r, scale=s_r, bias=None),
                    g1_r.at[:, pl.ds(c0, half_tn1)],
                    p1_r,
                    acc_r,
                    meta,
                    cfgs=cfgs1,
                    gs_gate_ref=gs_g_r,
                    gs_up_ref=gs_u_r,
                    scatter_mode=True,
                    _k_id=0,
                    _num_k=1,
                    _gm_id=0,
                    _n_id=n_id,
                    _num_n=2,
                    fused_g2_amax_slot=(None if static_g2_s > 0.0 else 1),
                    prequant_lhs_fp8_ref=pq_fp8_r,
                    prequant_lhs_scale_ref=pq_sc_r,
                    skip_row_mask=skip_mask,
                    lhs_3d_vmem_ref=x3d_r,
                )
            inner_kernel(
                None,
                WeightsRef(weight=w2_r, scale=s2_r, bias=None),
                out_r,
                p1_r,
                acc_r,
                meta,
                cfgs=cfgs2,
                gs_gate_ref=gs_w2_r,
                gs_up_ref=gs_w2_r,
                scatter_mode=True,
                _k_id=0,
                _num_k=1,
                _gm_id=0,
                _n_id=0,
                _num_n=1,
                fused_g2_amax_slot=(None if static_g2_s > 0.0 else 1),
                prequant_lhs_fp8_ref=(
                    g1_r if static_g2_s > 0.0 else pq_fp8_r.at[:, :k2]
                ),
                prequant_lhs_scale_ref=pq_sc_r,
                skip_row_mask=skip_mask,
                lhs_2d_vmem_ref=(None if static_g2_s > 0.0 else g1_r),
                prequant_lhs_ready=(static_g2_s > 0.0),
                skip_lhs_scale_mul=(static_g2_s > 0.0),
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
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "8",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": ds_mode,
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "3",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                },
            ),
        ):
            call = pl.pallas_call(
                _ker,
                out_shape=jax.ShapeDtypeStruct((m_rows, n2), jnp.bfloat16),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 7
                + [pl.BlockSpec(memory_space=pltpu.SMEM)] * 3,
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((m_rows, n1_half), g1_dtype),
                    pltpu.VMEM((m_rows, k1), jnp.float8_e4m3fn),
                    pltpu.VMEM(
                        (1 if static_g2_s > 0.0 else 2, m_rows, 128), jnp.float32
                    ),
                    pltpu.VMEM((8, tile_n1_g), jnp.float32),
                    pltpu.VMEM((m_rows, tile_n1_g), jnp.float32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                ],
                interpret=eager_interp,
            )
            return np.asarray(
                call(
                    lhs_3d,
                    w1_t0,
                    s1_t0,
                    w1_t1,
                    s1_t1,
                    w2_u32,
                    s2_bf16,
                    gs_gate,
                    gs_up,
                    gs_w2,
                ),
                dtype=np.float32,
            )

    def _run_pair_same_rng(
        lhs_scale_f: float,
        *,
        static_g2_s: float,
        ds_mode: str = "1",
        skip_mask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        nonlocal rng
        saved_state = rng.bit_generator.state
        y_dyn = _run_g2_case(
            lhs_scale_f, static_g2_s=0.0, ds_mode=ds_mode, skip_mask=skip_mask
        )
        rng.bit_generator.state = saved_state
        y_st = _run_g2_case(
            lhs_scale_f, static_g2_s=static_g2_s, ds_mode=ds_mode, skip_mask=skip_mask
        )
        return y_dyn, y_st

    for ds_mode in ("1", "2"):
        y_dyn, y_st = _run_pair_same_rng(0.25, static_g2_s=4.0, ds_mode=ds_mode)
        assert np.all(np.isfinite(y_st))
        cos = float(
            np.dot(y_dyn.ravel(), y_st.ravel())
            / (np.linalg.norm(y_dyn.ravel()) * np.linalg.norm(y_st.ravel()) + 1e-30)
        )
        assert cos > 0.99, f"ds={ds_mode} static gmm2 cos too low: {cos}"

    _, y_st_mask = _run_pair_same_rng(0.25, static_g2_s=2.0, ds_mode="1", skip_mask=False)
    assert np.all(np.isfinite(y_st_mask))
    assert np.all(y_st_mask[-19:] == 0.0)

    # Large activations saturating +-448 before FP8 cast must stay finite (no IEEE FP8 NaN).
    _, y_clip = _run_pair_same_rng(32.0, static_g2_s=16.0, ds_mode="1")
    assert np.all(np.isfinite(y_clip))
    _, y_clip_ds2 = _run_pair_same_rng(32.0, static_g2_s=16.0, ds_mode="2")
    assert np.all(np.isfinite(y_clip_ds2))

    base_env = {
        "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "3",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
        "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "8",
        "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": "1",
        "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
        "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
        "SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS": "12",
        "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "6",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS": "8.0",
        "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "2",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "256,4096,4096,4096,4096,2,1",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
        "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
        "SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS": "-1",
    }

    def _compile_gmm2_aot(
        topo_name: str,
        ep_size: int,
        e_local: int,
        gmm2_s: str,
        *,
        with_gs: bool,
        extra_env: dict[str, str] | None = None,
    ) -> tuple[int, str]:
        topo = topologies.get_topology_desc(platform="tpu", topology_name=topo_name)
        mesh = Mesh(np.array(topo.devices), ("model",))
        e_global = ep_size * e_local
        n_slots = e_local + (24576 * 8 + 255) // 256
        env_map = {
            **base_env,
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_GMM2_LHS": gmm2_s,
            **(extra_env or {}),
        }
        with patch.dict(os.environ, env_map):
            jax.clear_caches()

            def _fn(h, w1, w1s, w2, w2s, w1gs, w2gs, gs, li, g_st, go, mn, pnw):
                return gmm_v2_fused_rs(
                    h,
                    w1,
                    w2,
                    gs,
                    li,
                    li,
                    w1_scale=w1s,
                    w2_scale=w2s,
                    w1_global_scale=(w1gs if with_gs else None),
                    w2_global_scale=(w2gs if with_gs else None),
                    act_fn="silu",
                    group_offset=go,
                    group_start=g_st,
                    output_size=24576,
                    topk_indices=jnp.zeros((1,), jnp.int32),
                    ep_size=ep_size,
                    ep_axis_name="model",
                    max_num_gm=mn[0],
                    caller_tile_m=256,
                    top_k=8,
                    post_expert_norm_weight=pnw,
                    block_scatter=True,
                )

            in_sp = (
                P(),
                P("model"),
                P("model"),
                P("model"),
                P("model"),
                P("model"),
                P("model"),
                P(),
                P(),
                P(),
                P("model"),
                P(),
                P(),
            )
            sd = jax.ShapeDtypeStruct
            sh_r = NamedSharding(mesh, P())
            sh_m = NamedSharding(mesh, P("model"))
            specs = [
                sd((24576, 32, 128), jnp.float8_e4m3fn, sharding=sh_r),
                sd((e_global, 4096, 8192), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, 8, 8192), jnp.bfloat16, sharding=sh_m),
                sd((e_global, 4096, 4096), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, 8, 4096), jnp.bfloat16, sharding=sh_m),
                sd((e_global, 2), jnp.float32, sharding=sh_m),
                sd((e_global,), jnp.float32, sharding=sh_m),
                sd((e_global,), jnp.int32, sharding=sh_r),
                sd((196608,), jnp.int32, sharding=sh_r),
                sd((4 + 2 * n_slots,), jnp.int32, sharding=sh_r),
                sd((ep_size,), jnp.int32, sharding=sh_m),
                sd((1,), jnp.int32, sharding=sh_r),
                sd((4096,), jnp.bfloat16, sharding=sh_r),
            ]
            lowered = (
                jax.jit(
                    jax.shard_map(
                        _fn,
                        mesh=mesh,
                        in_specs=in_sp,
                        out_specs=P("model", None, None),
                        check_vma=False,
                    )
                )
                .trace(*specs)
                .lower()
            )
            hlo = lowered.as_text("stablehlo")
            compiled = lowered.compile()
            code_b = int(
                getattr(compiled.memory_analysis(), "generated_code_size_in_bytes", -1)
            )
            return code_b, hlo

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        try:
            code_base, hlo_base = _compile_gmm2_aot("tpu7x=2x2x1", 8, 36, "0.0", with_gs=True)
            code_g2s4, hlo_g2s4 = _compile_gmm2_aot("tpu7x=2x2x1", 8, 36, "4.0", with_gs=True)
            assert "-g2s" not in hlo_base
            assert "-g2s4" in hlo_g2s4 and "-pql6-slh8" in hlo_g2s4
            assert 0 < code_g2s4 < code_base and code_g2s4 <= 3_385_344, (
                f"Expected g2s4 ({code_g2s4} B) < base ({code_base} B) and <= 3,385,344 B"
            )

            # None global scales must still pass folded S_gmm2 and 1/S_gmm2 safely.
            code_nogs, hlo_nogs = _compile_gmm2_aot("tpu7x=2x2x1", 8, 36, "2.0", with_gs=False)
            assert "-g2s2" in hlo_nogs
            assert 0 < code_nogs <= 3_385_344

            code_ep16, hlo_ep16 = _compile_gmm2_aot("tpu7x=2x2x2", 16, 28, "4.0", with_gs=True)
            assert "-g2s4" in hlo_ep16
            assert 0 < code_ep16 <= 3_385_344

            # Guardrail: disabled safely when eo > 0 or non-single-pass fa=1, ds=0.
            _, hlo_guard = _compile_gmm2_aot(
                "tpu7x=2x2x1",
                8,
                36,
                "4.0",
                with_gs=True,
                extra_env={
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": "0",
                },
            )
            assert "-g2s" not in hlo_guard
        finally:
            jax.clear_caches()


def test_scoped_w4a8_dual_qbs256_and_block256_gmm2_gate() -> None:
    """Verify rhs_qbs=256 single-strip delegation parity, qbs=2048 preservation, and BLOCK256 g2s gate."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from sglang_tpu.moe.pallas.fused_rs_muse import prefill_gmm_opts as opts
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        gmm_v2_fused_rs,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_v2_gather_scatter import (
        Dimensions,
        GmmConfigs,
        InputConfigs,
        MetadataRef,
        TileSizes,
        WeightsRef,
        inner_kernel,
    )

    calls: list[tuple[int, int, int, bool, bool]] = []

    def _fake_strip(**kw):
        calls.append(
            (
                kw["rhs_qbs"],
                kw["start_n"],
                kw["end_n"],
                bool(kw["mh_outer"]),
                bool(kw["hoist_rhs_scale"]),
            )
        )
        return kw["start_n"]

    with patch.object(opts, "scoped_w4a8_strip_acc", side_effect=_fake_strip):
        ag, au = opts.scoped_w4a8_dual_strip_acc(
            w_weight_ref=None,
            w_scale_ref=None,
            g_s=0,
            g_e=256,
            u_s=512,
            u_e=768,
            tile_m=256,
            tile_k=2048,
            q_block_size=2048,
            rhs_qbs=256,
            packing=8,
            rhs_quant_dtype=jnp.float4_e2m1fn,
            acc_dtype=jnp.float32,
            use_bf16_inner=True,
            prequant_lhs_fp8_ref=None,
            has_prequant_lhs=True,
            hoisted_lq_blocks=[],
            load_lhs_scale_strip_fn=None,
            zero_spill_sub_m=False,
            mh_outer=False,
            hoist_rhs_scale=True,
        )
        assert (ag, au) == (0, 512)
        assert calls == [
            (256, 0, 256, False, True),
            (256, 512, 768, False, True),
        ]
        calls.clear()
        opts.scoped_w4a8_dual_strip_acc(
            w_weight_ref=None,
            w_scale_ref=None,
            g_s=0,
            g_e=256,
            u_s=512,
            u_e=768,
            tile_m=256,
            tile_k=2048,
            q_block_size=2048,
            rhs_qbs=2048,
            packing=8,
            rhs_quant_dtype=jnp.float4_e2m1fn,
            acc_dtype=jnp.float32,
            use_bf16_inner=True,
            prequant_lhs_fp8_ref=None,
            has_prequant_lhs=True,
            hoisted_lq_blocks=[],
            load_lhs_scale_strip_fn=None,
            zero_spill_sub_m=False,
            mh_outer=False,
            hoist_rhs_scale=True,
        )
        assert calls == [
            (2048, 0, 256, True, True),
            (2048, 512, 768, True, True),
        ]

    # InterpretParams 0-ULP parity at rhs_qbs=256: fa=3 + sw=8 matches fa=1 + sw=8 bit-identically.
    rng = np.random.default_rng(20261201)
    m_rows, k1, tile_n1_g, n1_half, k2, n2 = 256, 2048, 512, 512, 512, 256
    w1_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    s1_bf16 = jnp.asarray(
        np.exp(rng.normal(-2.2, 0.2, size=(k1 // 256, n1_half * 2))).astype(
            np.float32
        ),
        jnp.bfloat16,
    )
    w2_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32)
    )
    s2_bf16 = jnp.asarray(
        np.exp(rng.normal(-2.2, 0.2, size=(k2 // 256, n2))).astype(np.float32),
        jnp.bfloat16,
    )
    half_tn1 = tile_n1_g // 2
    w1_tiles, s1_tiles = [], []
    for n_id in range(n1_half // half_tn1):
        c0 = n_id * half_tn1
        _g_sl = slice(c0, c0 + half_tn1)
        _u_sl = slice(n1_half + c0, n1_half + c0 + half_tn1)
        w1_tiles.append(jnp.concatenate([w1_u32[:, _g_sl], w1_u32[:, _u_sl]], axis=1))
        s1_tiles.append(jnp.concatenate([s1_bf16[:, _g_sl], s1_bf16[:, _u_sl]], axis=1))
    w1_t0, w1_t1 = w1_tiles
    s1_t0, s1_t1 = s1_tiles
    lhs_3d = jax.device_put(
        jnp.asarray(
            rng.standard_normal((m_rows, k1 // 128, 128)).astype(np.float32) * 0.25,
            jnp.bfloat16,
        )
    )
    gs_gate = jnp.asarray([1.05], jnp.float32)
    gs_up = jnp.asarray([0.95], jnp.float32)
    gs_w2 = jnp.asarray([1.1], jnp.float32)
    cfgs1 = GmmConfigs(
        dims=Dimensions(
            size_m=8192,
            size_k=k1,
            size_n=n1_half * 2,
            size_group=1,
            size_lhs_group=1,
            size_lhs_sublane=8,
        ),
        tiles=TileSizes(tile_m=m_rows, tile_k=k1, tile_n=tile_n1_g),
        lhs_cfgs=InputConfigs(
            quant_dtype=jnp.float8_e4m3fn.dtype,
            quant_block_size=k1,
            dtype=jnp.bfloat16,
        ),
        rhs_cfgs=InputConfigs(
            quant_dtype=jnp.float4_e2m1fn.dtype,
            quant_block_size=256,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k1 // 256,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act="silu",
    )
    cfgs2 = GmmConfigs(
        dims=Dimensions(
            size_m=8192,
            size_k=k2,
            size_n=n2,
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
            quant_block_size=256,
            dtype=jnp.float4_e2m1fn,
            has_bias=False,
            has_scale=True,
            packing=8,
            num_quant_blocks=k2 // 256,
        ),
        out_dtype=jnp.bfloat16,
        acc_dtype=jnp.float32,
        zero_init=False,
        fuse_act=None,
    )
    eager_interp = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )

    def _run_fa(fa_mode: str) -> np.ndarray:
        def _ker(x3d_r, w1r0, s1r0, w1r1, s1r1, w2_r, s2_r, gs_g_r, gs_u_r, gs_w2_r, out_r, g1_r, pq_fp8_r, pq_sc_r, p1_r, acc_r, gid_r, moff_r):
            gid_r[0] = 0
            moff_r[0] = 0
            moff_r[1] = m_rows
            meta = MetadataRef(gm_id_to_group_id=gid_r, gm_id_to_m_offset=moff_r)
            for n_id, (wr, sr) in enumerate(((w1r0, s1r0), (w1r1, s1r1))):
                c0 = n_id * half_tn1
                inner_kernel(
                    None, WeightsRef(weight=wr, scale=sr, bias=None), g1_r.at[:, pl.ds(c0, half_tn1)], p1_r, acc_r, meta,
                    cfgs=cfgs1, gs_gate_ref=gs_g_r, gs_up_ref=gs_u_r, scatter_mode=True,
                    _k_id=0, _num_k=1, _gm_id=0, _n_id=n_id, _num_n=2,
                    fused_g2_amax_slot=None, prequant_lhs_fp8_ref=pq_fp8_r,
                    prequant_lhs_scale_ref=pq_sc_r, skip_row_mask=True, lhs_3d_vmem_ref=x3d_r,
                )
            inner_kernel(
                None, WeightsRef(weight=w2_r, scale=s2_r, bias=None), out_r, p1_r, acc_r, meta,
                cfgs=cfgs2, gs_gate_ref=gs_w2_r, gs_up_ref=gs_w2_r, scatter_mode=True,
                _k_id=0, _num_k=1, _gm_id=0, _n_id=0, _num_n=1,
                fused_g2_amax_slot=None, prequant_lhs_fp8_ref=g1_r,
                prequant_lhs_scale_ref=pq_sc_r, skip_row_mask=True,
                lhs_2d_vmem_ref=None, prequant_lhs_ready=True, skip_lhs_scale_mul=True,
            )

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
            patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "8",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": "1",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": fa_mode,
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                },
            ),
        ):
            call = pl.pallas_call(
                _ker,
                out_shape=jax.ShapeDtypeStruct((m_rows, n2), jnp.bfloat16),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 7 + [pl.BlockSpec(memory_space=pltpu.SMEM)] * 3,
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((m_rows, n1_half), jnp.float8_e4m3fn),
                    pltpu.VMEM((m_rows, k1), jnp.float8_e4m3fn),
                    pltpu.VMEM((1, m_rows, 128), jnp.float32),
                    pltpu.VMEM((8, tile_n1_g), jnp.float32),
                    pltpu.VMEM((m_rows, tile_n1_g), jnp.float32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                ],
                interpret=eager_interp,
            )
            return np.asarray(call(lhs_3d, w1_t0, s1_t0, w1_t1, s1_t1, w2_u32, s2_bf16, gs_gate, gs_up, gs_w2), np.float32)

    y_fa1 = _run_fa("1")
    y_fa3 = _run_fa("3")
    assert np.all(np.isfinite(y_fa1)) and np.all(np.isfinite(y_fa3))
    assert np.array_equal(y_fa3, y_fa1)

    # Verify g2s4 activates on BLOCK256_SCALES=1 prefill when K256_PREFILL_TILES=0, and disables when =1.
    def _compile_nb(nb: int, extra_env: dict[str, str], *, lhs_dtype=None) -> tuple[int, str]:
        topo = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x=2x2x1")
        mesh = Mesh(np.array(topo.devices), ("model",))
        ep_size, e_local = 8, 36
        e_global = ep_size * e_local
        n_slots = e_local + (24576 * 8 + 255) // 256
        h_dt = jnp.float8_e4m3fn if lhs_dtype is None else lhs_dtype
        env_map = {
            "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
            "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
            "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "3",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
            "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
            "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
            "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "8",
            "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": "1",
            "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
            "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
            "SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS": "12",
            "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "6",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS": "8.0",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_GMM2_LHS": "4.0",
            "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "2",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "256,4096,4096,4096,4096,2,1",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
            "SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS": "-1",
            **extra_env,
        }
        with patch.dict(os.environ, env_map):
            jax.clear_caches()

            def _fn(h, w1, w1s, w2, w2s, w1gs, w2gs, gs, li, g_st, go, mn, pnw):
                return gmm_v2_fused_rs(
                    h, w1, w2, gs, li, li,
                    w1_scale=w1s, w2_scale=w2s, w1_global_scale=w1gs, w2_global_scale=w2gs,
                    act_fn="silu", group_offset=go, group_start=g_st, output_size=24576,
                    topk_indices=jnp.zeros((1,), jnp.int32), ep_size=ep_size, ep_axis_name="model",
                    max_num_gm=mn[0], caller_tile_m=256, top_k=8, post_expert_norm_weight=pnw, block_scatter=True,
                )

            in_sp = (
                P(), P("model"), P("model"), P("model"), P("model"), P("model"), P("model"),
                P(), P(), P(), P("model"), P(), P(),
            )
            sd = jax.ShapeDtypeStruct
            sh_r, sh_m = NamedSharding(mesh, P()), NamedSharding(mesh, P("model"))
            specs = [
                sd((24576, 32, 128), h_dt, sharding=sh_r),
                sd((e_global, 4096, 8192), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, nb, 8192), jnp.bfloat16, sharding=sh_m),
                sd((e_global, 4096, 4096), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, nb, 4096), jnp.bfloat16, sharding=sh_m),
                sd((e_global, 2), jnp.float32, sharding=sh_m),
                sd((e_global,), jnp.float32, sharding=sh_m),
                sd((e_global,), jnp.int32, sharding=sh_r),
                sd((196608,), jnp.int32, sharding=sh_r),
                sd((4 + 2 * n_slots,), jnp.int32, sharding=sh_r),
                sd((ep_size,), jnp.int32, sharding=sh_m),
                sd((1,), jnp.int32, sharding=sh_r),
                sd((4096,), jnp.bfloat16, sharding=sh_r),
            ]
            lowered = jax.jit(jax.shard_map(_fn, mesh=mesh, in_specs=in_sp, out_specs=P("model", None, None), check_vma=False)).trace(*specs).lower()
            hlo = lowered.as_text("stablehlo")
            compiled = lowered.compile()
            return int(getattr(compiled.memory_analysis(), "generated_code_size_in_bytes", -1)), hlo

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        try:
            code_u, hlo_u = _compile_nb(
                16,
                {
                    "SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES": "1",
                    "SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES": "1",
                    "SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_TILES": "0",
                },
            )
            assert "-g2s4" in hlo_u and "-fa3" in hlo_u and "-sw8" in hlo_u and "-b256" not in hlo_u
            assert code_u > 0

            _, hlo_k256_pre = _compile_nb(
                16,
                {
                    "SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES": "1",
                    "SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES": "1",
                    "SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_TILES": "1",
                },
                lhs_dtype=jnp.bfloat16,
            )
            assert "-g2s" not in hlo_k256_pre and "-b256" in hlo_k256_pre

            code_2k, hlo_2k = _compile_nb(2, {})
            assert "-g2s4" in hlo_2k and "-fa3" in hlo_2k
            assert 0 < code_2k <= 3_385_344
        finally:
            jax.clear_caches()


def test_w4a8_k_chain_scaled_fp8_interpret_and_aot() -> None:
    """Verify Wave-14 K-chained Scaled-FP8 validator, eager race/NaN/mask/zero-scale parity, and TPU7x AOT."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        _select_fused_rs_block_sizes,
        gmm_v2_fused_rs,
    )
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
        resolve_w4a8_k_chain,
    )

    assert resolve_w4a8_k_chain(None) == 0
    assert resolve_w4a8_k_chain(0) == 0
    for valid_kc in (512, 1024, 2048, 4096):
        assert resolve_w4a8_k_chain(valid_kc) == valid_kc
    for bad_kc in (-1, 128, 256, 300, 1536, 8192):
        with pytest.raises(ValueError):
            resolve_w4a8_k_chain(bad_kc)

    rng = np.random.default_rng(20261205)
    m_rows, k1, n1_half, k2, n2 = 256, 2048, 2048, 2048, 256
    tile_n1_g = n1_half * 2
    # Negative scales and a zero-scale/dead slice exercise sign & zero-col safety.
    w1_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k1 // 8, n1_half * 2), dtype=np.uint32)
    )
    s1_np = np.exp(rng.normal(-2.2, 0.2, size=(k1 // 256, n1_half * 2))).astype(
        np.float32
    )
    s1_np[:, :16] = 0.0
    s1_np[0::2, 16:32] *= -1.0
    s1_np[:, n1_half : n1_half + 16] = 0.0
    s1_bf16 = jnp.asarray(s1_np, jnp.bfloat16)

    w2_u32 = jnp.asarray(
        rng.integers(0, 2**31 - 1, size=(k2 // 8, n2), dtype=np.uint32)
    )
    s2_np = np.exp(rng.normal(-2.2, 0.2, size=(k2 // 256, n2))).astype(np.float32)
    s2_np[:, :16] = 0.0
    s2_np[0::2, 16:32] *= -1.0
    s2_bf16 = jnp.asarray(s2_np, jnp.bfloat16)

    lhs_3d = jax.device_put(
        jnp.asarray(
            rng.standard_normal((m_rows, k1 // 128, 128)).astype(np.float32) * 0.25,
            jnp.bfloat16,
        )
    )
    gs_gate = jnp.asarray([1.05], jnp.float32)
    gs_up = jnp.asarray([0.95], jnp.float32)
    gs_w2 = jnp.asarray([1.1], jnp.float32)

    def _mk_cfgs(qbs_lhs: int, with_g1: bool, *, rhs_qbs: int = 256):
        n_cols = tile_n1_g if with_g1 else n2
        return GmmConfigs(
            dims=Dimensions(
                size_m=8192,
                size_k=k1 if with_g1 else k2,
                size_n=n_cols,
                size_group=1,
                size_lhs_group=1,
                size_lhs_sublane=8,
            ),
            tiles=TileSizes(tile_m=m_rows, tile_k=k1 if with_g1 else k2, tile_n=n_cols),
            lhs_cfgs=InputConfigs(
                quant_dtype=jnp.float8_e4m3fn.dtype,
                quant_block_size=qbs_lhs,
                dtype=jnp.bfloat16,
            ),
            rhs_cfgs=InputConfigs(
                quant_dtype=jnp.float4_e2m1fn.dtype,
                quant_block_size=rhs_qbs,
                dtype=jnp.float4_e2m1fn,
                has_bias=False,
                has_scale=True,
                packing=8,
                num_quant_blocks=(k1 if with_g1 else k2) // rhs_qbs,
            ),
            out_dtype=jnp.bfloat16,
            acc_dtype=jnp.float32,
            zero_init=False,
            fuse_act="silu" if with_g1 else None,
        )

    eager_interp = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )

    def _run_e2e_inner(
        kc_str: str,
        *,
        g2s_ready: bool,
        skip_mask: bool,
        rhs_qbs: int = 256,
        kc_override: int | None = None,
    ) -> np.ndarray:
        c1 = _mk_cfgs(k1, True, rhs_qbs=rhs_qbs)
        c2 = _mk_cfgs(k2 if g2s_ready else 1024, False, rhs_qbs=rhs_qbs)

        def _ker(x3d_r, w1_r, s1_r, w2_r, s2_r, gs_g_r, gs_u_r, gs_w2_r, out_r, g1_r, pq_fp8_r, pq_sc_r, p1_r, acc_r, gid_r, moff_r):
            gid_r[0] = 0
            moff_r[0] = 0 if skip_mask else 5
            moff_r[1] = m_rows if skip_mask else 192
            meta = MetadataRef(gm_id_to_group_id=gid_r, gm_id_to_m_offset=moff_r)
            inner_kernel(
                None, WeightsRef(weight=w1_r, scale=s1_r, bias=None), g1_r, p1_r, acc_r, meta,
                cfgs=c1, gs_gate_ref=gs_g_r, gs_up_ref=gs_u_r, scatter_mode=True,
                _k_id=0, _num_k=1, _gm_id=0, _n_id=0, _num_n=1,
                fused_g2_amax_slot=None, prequant_lhs_fp8_ref=pq_fp8_r,
                prequant_lhs_scale_ref=pq_sc_r, skip_row_mask=skip_mask, lhs_3d_vmem_ref=x3d_r,
                w4a8_k_chain_override=kc_override,
            )
            inner_kernel(
                None, WeightsRef(weight=w2_r, scale=s2_r, bias=None), out_r, p1_r, acc_r, meta,
                cfgs=c2, gs_gate_ref=gs_w2_r, gs_up_ref=gs_w2_r, scatter_mode=True,
                _k_id=0, _num_k=1, _gm_id=0, _n_id=0, _num_n=1,
                fused_g2_amax_slot=None,
                prequant_lhs_fp8_ref=g1_r if g2s_ready else pq_fp8_r,
                prequant_lhs_scale_ref=pq_sc_r, skip_row_mask=skip_mask,
                lhs_2d_vmem_ref=None if g2s_ready else g1_r,
                prequant_lhs_ready=g2s_ready,
                skip_lhs_scale_mul=g2s_ready,
                w4a8_k_chain_override=kc_override,
            )

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
            patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "8",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_W4A8_K_CHAIN": kc_str,
                    "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": "1",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "3",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
                },
            ),
        ):
            n_g2_lhs_b = max(1, k2 // c2.lhs_cfgs.quant_block_size)
            s1_in = s1_bf16 if rhs_qbs == 256 else s1_bf16[:: (rhs_qbs // 256)]
            s2_in = s2_bf16 if rhs_qbs == 256 else s2_bf16[:: (rhs_qbs // 256)]
            call = pl.pallas_call(
                _ker,
                out_shape=jax.ShapeDtypeStruct((m_rows, n2), jnp.bfloat16),
                in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 5 + [pl.BlockSpec(memory_space=pltpu.SMEM)] * 3,
                out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                scratch_shapes=[
                    pltpu.VMEM((m_rows, n1_half), jnp.float8_e4m3fn if g2s_ready else jnp.bfloat16),
                    pltpu.VMEM((m_rows, max(k1, k2)), jnp.float8_e4m3fn),
                    pltpu.VMEM((n_g2_lhs_b, m_rows, 128), jnp.float32),
                    pltpu.VMEM((8, tile_n1_g), jnp.float32),
                    pltpu.VMEM((m_rows, tile_n1_g), jnp.float32),
                    pltpu.SMEM((1,), jnp.int32),
                    pltpu.SMEM((2,), jnp.int32),
                ],
                interpret=eager_interp,
            )
            return np.asarray(call(lhs_3d, w1_u32, s1_in, w2_u32, s2_in, gs_gate, gs_up, gs_w2), np.float32)

    # 1) Hot path: skip_lhs_scale_mul=True, full k_chain=4096 (clamps to 2048), multi-chunk 1024, override=0
    y_ref = _run_e2e_inner("0", g2s_ready=True, skip_mask=True)
    y_ov0 = _run_e2e_inner("4096", g2s_ready=True, skip_mask=True, kc_override=0)
    np.testing.assert_array_equal(y_ov0, y_ref)
    y_kc4096 = _run_e2e_inner("4096", g2s_ready=True, skip_mask=True)
    y_kc1024 = _run_e2e_inner("1024", g2s_ready=True, skip_mask=True)
    for y_k in (y_kc4096, y_kc1024):
        assert np.all(np.isfinite(y_k))
        assert np.all(y_k[:, :16] == 0.0)
        cos_v = float(np.dot(y_ref.ravel(), y_k.ravel()) / (np.linalg.norm(y_ref.ravel()) * np.linalg.norm(y_k.ravel())))
        assert cos_v > 0.997, f"k_chain cos={cos_v}"

    # 1b) qbs=512 (nb=8) cleanly bypasses k_chain even if env is "4096"
    y_q512_ref = _run_e2e_inner("0", g2s_ready=True, skip_mask=True, rhs_qbs=512)
    y_q512_kc = _run_e2e_inner("4096", g2s_ready=True, skip_mask=True, rhs_qbs=512)
    np.testing.assert_array_equal(y_q512_kc, y_q512_ref)

    # 2) Dynamic-LHS & row-mask path: skip_lhs_scale_mul=False, qbs_lhs=1024, tail mask [5:192]
    y_dyn_ref = _run_e2e_inner("0", g2s_ready=False, skip_mask=False)
    y_dyn_kc2048 = _run_e2e_inner("2048", g2s_ready=False, skip_mask=False)
    assert np.all(np.isfinite(y_dyn_kc2048))
    assert np.all(y_dyn_kc2048[:5, :] == 0.0) and np.all(y_dyn_kc2048[192:, :] == 0.0)
    assert np.all(y_dyn_kc2048[:, :16] == 0.0)
    cos_dyn = float(np.dot(y_dyn_ref.ravel(), y_dyn_kc2048.ravel()) / (np.linalg.norm(y_dyn_ref.ravel()) * np.linalg.norm(y_dyn_kc2048.ravel())))
    assert cos_dyn > 0.997, f"k_chain dyn cos={cos_dyn}"

    # 3) TPU7x 2x2x1 (TP=8) & 2x2x2 (TP=16) AOT HLO tag, W2_FP8_CACHE suppression & IMEM check
    def _compile_topo(
        topo_name: str,
        ep_size: int,
        e_local: int,
        extra_env: dict[str, str],
        *,
        nb: int = 16,
    ) -> tuple[int, str]:
        topo = topologies.get_topology_desc(platform="tpu", topology_name=topo_name)
        mesh = Mesh(np.array(topo.devices), ("model",))
        e_global = ep_size * e_local
        n_slots = e_local + (24576 * 8 + 255) // 256
        env_map = {
            "SGLANG_TPU_MUSE_KERNEL_FLATTEN_BS_META": "1",
            "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
            "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "3",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
            "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
            "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW": "3",
            "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "8",
            "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": "1",
            "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "6",
            "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
            "SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS": "12",
            "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "6",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS": "8.0",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_GMM2_LHS": "4.0",
            "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "2",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "256,4096,4096,4096,4096,2,1",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "6.0",
            "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS": "4",
            "SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS": "-1",
            "SGLANG_TPU_MUSE_KERNEL_BLOCK256_SCALES": "1" if nb == 16 else "0",
            "SGLANG_TPU_MUSE_KERNEL_COMPACT_K256_SCALES": "1" if nb == 16 else "0",
            "SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_TILES": "0",
            **({"SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": str(4096 // nb)} if nb != 16 else {}),
            **extra_env,
        }
        with patch.dict(os.environ, env_map):
            jax.clear_caches()

            def _fn(h, w1, w1s, w2, w2s, w1gs, w2gs, gs, li, g_st, go, mn, pnw):
                return gmm_v2_fused_rs(
                    h, w1, w2, gs, li, li,
                    w1_scale=w1s, w2_scale=w2s, w1_global_scale=w1gs, w2_global_scale=w2gs,
                    act_fn="silu", group_offset=go, group_start=g_st, output_size=24576,
                    topk_indices=jnp.zeros((1,), jnp.int32), ep_size=ep_size, ep_axis_name="model",
                    max_num_gm=mn[0], caller_tile_m=256, top_k=8, post_expert_norm_weight=pnw, block_scatter=True,
                )

            in_sp = (
                P(), P("model"), P("model"), P("model"), P("model"), P("model"), P("model"),
                P(), P(), P(), P("model"), P(), P(),
            )
            sd = jax.ShapeDtypeStruct
            sh_r, sh_m = NamedSharding(mesh, P()), NamedSharding(mesh, P("model"))
            specs = [
                sd((24576, 32, 128), jnp.float8_e4m3fn, sharding=sh_r),
                sd((e_global, 4096, 8192), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, nb, 8192), jnp.bfloat16, sharding=sh_m),
                sd((e_global, 4096, 4096), jnp.float4_e2m1fn, sharding=sh_m),
                sd((e_global, nb, 4096), jnp.bfloat16, sharding=sh_m),
                sd((e_global, 2), jnp.float32, sharding=sh_m),
                sd((e_global,), jnp.float32, sharding=sh_m),
                sd((e_global,), jnp.int32, sharding=sh_r),
                sd((196608,), jnp.int32, sharding=sh_r),
                sd((4 + 2 * n_slots,), jnp.int32, sharding=sh_r),
                sd((ep_size,), jnp.int32, sharding=sh_m),
                sd((1,), jnp.int32, sharding=sh_r),
                sd((4096,), jnp.bfloat16, sharding=sh_r),
            ]
            lowered = jax.jit(jax.shard_map(_fn, mesh=mesh, in_specs=in_sp, out_specs=P("model", None, None), check_vma=False)).trace(*specs).lower()
            hlo = lowered.as_text("stablehlo")
            compiled = lowered.compile()
            return int(getattr(compiled.memory_analysis(), "generated_code_size_in_bytes", -1)), hlo

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        try:
            code_base8, hlo_base8 = _compile_topo("tpu7x=2x2x1", 8, 36, {})
            code_kc8, hlo_kc8 = _compile_topo(
                "tpu7x=2x2x1", 8, 36,
                {
                    "SGLANG_TPU_MUSE_KERNEL_GMM_W4A8_K_CHAIN": "4096",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE": "1",
                },
            )
            assert "-kc" not in hlo_base8
            assert "-kc4096" in hlo_kc8 and "-w2c1" not in hlo_kc8
            assert 0 < code_kc8 < code_base8

            # Wave-15: nb=8 (qbs=512) cleanly suppresses k_chain, preserves W2_FP8_CACHE,
            # and PREFILL_GMM_TILES="" + SLH8 + WLA=2 auto-selects full out-N (256, 4096..2, 1).
            code_nb8, hlo_nb8 = _compile_topo(
                "tpu7x=2x2x1",
                8,
                36,
                {
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_W4A8_K_CHAIN": "4096",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE": "1",
                },
                nb=8,
            )
            assert "-wla2" in hlo_nb8 and "-w2c1" in hlo_nb8 and "-kc" not in hlo_nb8
            assert 0 < code_nb8 <= 3_385_344

            # Wave-15: half-N n2048 (2,2) on nb=16 cleanly suppresses k_chain (no 4.84 MiB IMEM bloat).
            code_n2048, hlo_n2048 = _compile_topo(
                "tpu7x=2x2x1",
                8,
                36,
                {
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "256,4096,2048,4096,2048,2,2",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_W4A8_K_CHAIN": "4096",
                },
            )
            assert "-kc" not in hlo_n2048 and code_n2048 > 0

            code_nb2, hlo_nb2 = _compile_topo(
                "tpu7x=2x2x1", 8, 36,
                {"SGLANG_TPU_MUSE_KERNEL_GMM_W4A8_K_CHAIN": "4096"},
                nb=2,
            )
            assert "-kc" not in hlo_nb2 and 0 < code_nb2 <= 3_385_344

            code_kc16, hlo_kc16 = _compile_topo(
                "tpu7x=2x2x2",
                16,
                20,
                {
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_W4A8_K_CHAIN": "4096",
                },
            )
            assert "-wla2" in hlo_kc16 and "-kc4096" in hlo_kc16 and "-g2s4" in hlo_kc16
            assert code_kc16 > 0

            # Direct block-size selector guard checks across M / SLH / WLA / EO / qbs.
            base_bs_kw = dict(
                size_k1=4096,
                size_n1=8192,
                size_k2=4096,
                size_n2=4096,
                size_group=36,
                size_lhs_group=288,
                ep_size=8,
                out_dtype=jnp.bfloat16,
                w1_dtype=jnp.float4_e2m1fn,
                w2_dtype=jnp.float4_e2m1fn,
                is_quantized=True,
                act_fn="silu",
            )
            slh_wla2_env = {
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "",
                "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "6",
                "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS": "8.0",
                "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "2",
                "SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8": "0",
                "SGLANG_TPU_MUSE_KERNEL_K256_PREFILL_TILES": "0",
            }
            with patch.dict(os.environ, slh_wla2_env):
                for m_tok in (65536, 131072, 196608):
                    bs_auto = _select_fused_rs_block_sizes(
                        size_m=m_tok, quant_block_size=512, **base_bs_kw
                    )
                    assert (
                        bs_auto.tile_m,
                        bs_auto.tile_k1,
                        bs_auto.tile_n1,
                        bs_auto.tile_k2,
                        bs_auto.tile_n2,
                        bs_auto.num_w1_bufs,
                        bs_auto.num_w2_bufs,
                    ) == (256, 4096, 4096, 4096, 4096, 2, 1)
                assert (
                    _select_fused_rs_block_sizes(
                        size_m=32768, quant_block_size=512, **base_bs_kw
                    ).tile_m
                    == 128
                )
                assert (
                    _select_fused_rs_block_sizes(
                        size_m=65536, quant_block_size=16, **base_bs_kw
                    ).tile_n1
                    != 4096
                )
            with patch.dict(
                os.environ,
                {
                    **slh_wla2_env,
                    "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "4",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_LHS": "0",
                },
            ):
                assert (
                    _select_fused_rs_block_sizes(
                        size_m=65536, quant_block_size=512, **base_bs_kw
                    ).tile_n1
                    != 4096
                )
            with patch.dict(
                os.environ,
                {**slh_wla2_env, "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "0"},
            ):
                assert (
                    _select_fused_rs_block_sizes(
                        size_m=65536, quant_block_size=512, **base_bs_kw
                    ).tile_n1
                    != 4096
                )
            with patch.dict(
                os.environ,
                {**slh_wla2_env, "SGLANG_TPU_MUSE_KERNEL_GMM_EXPERT_OUTER_FP8": "3"},
            ):
                assert (
                    _select_fused_rs_block_sizes(
                        size_m=65536, quant_block_size=512, **base_bs_kw
                    ).tile_n1
                    != 4096
                )
        finally:
            jax.clear_caches()
