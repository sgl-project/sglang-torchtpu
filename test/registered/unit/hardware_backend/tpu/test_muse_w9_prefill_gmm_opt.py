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
    "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU",
    "SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW",
    "SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS",
    "SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS",
})


def test_w9_prefill_gmm_opt_flags_default_off() -> None:
    """Verify Wave-9 prefill GMM flags remain default-off."""
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_FLQ_ZERO_RMW.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_INTERLEAVE_RS.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PREFILL_ROW_BUCKETS.get() == 0


def test_no_bare_envfield_bool_eval_in_prefill_gmm_files() -> None:
    """Ensure every EnvField attribute checked here uses explicit `.get()`."""
    repo = pathlib.Path(__file__).resolve().parents[5]
    rel_paths = (
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_v2_gather_scatter.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_fused_rs_nodedup.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/prefill_gmm_opts.py",
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
    for ep_size, my_id, local_start in ((8, 3, 128), (16, 5, 0)):
        size_group = 6
        tile_m = 256
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
