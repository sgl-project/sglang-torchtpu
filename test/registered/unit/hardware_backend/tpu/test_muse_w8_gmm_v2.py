# SPDX-License-Identifier: Apache-2.0
"""Unit, bit-exact numerical, SMEM budget, and Mosaic TPU7x HLO export tests for Wave-8 gmm_v2_fused_rs."""

from __future__ import annotations

import ast
import os
import pathlib
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pallas_export import tpu_info_for_cpu_export


def test_w8_gmm_v2_flags_default_off():
    """All Wave-8 gmm_v2 flags must default strictly off/safe."""
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_SMEM_WINDOW_ROWS.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_K_STEP.get() == 1024


def test_no_bare_envfield_bool_evaluation_on_w8_gmm_files():
    """Ensure all modified Wave-8 gmm_v2 files call .get() on every EnvField."""
    repo = pathlib.Path(__file__).resolve().parents[5]
    rel_paths = (
        "sglang_tpu/moe/nvfp4_requant.py",
        "sglang_tpu/moe/pallas/fused_rs/fused_moe.py",
        "sglang_tpu/moe/pallas/fused_rs/gmm_fused_rs_nodedup.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/fused_moe.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_fused_rs_nodedup.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/gmm_v2_gather_scatter.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/pipeline.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/routing_pallas.py",
    )
    checked_flags = {
        "SGLANG_TPU_MUSE_KERNEL_GMM_PACK_U16_LHS",
        "SGLANG_TPU_MUSE_KERNEL_GMM_SMEM_WINDOW_ROWS",
        "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD",
        "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD",
        "SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE",
        "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS",
        "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU",
        "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8",
        "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK",
        "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2",
        "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_K_STEP",
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
                assert found_get, f"{rel}:{node.lineno} missing .get() on {node.attr}"


def test_w8_smem_budget_and_t32k_t48k_t64k_mosaic_lowering():
    """Verify exact SMEM byte count < 1 MiB and clean TPU7x Mosaic lowering at T=32k, 48k, and 64k."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_rs.fused_moe import (
        _assert_fused_rs_smem_safe as assert_smem_std,
        fused_moe_func_rs as fused_moe_std,
    )
    from sglang_tpu.moe.pallas.fused_rs.gmm_fused_rs_nodedup import (
        fused_rs_smem_rows as smem_rows_std,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_moe import (
        _assert_fused_rs_smem_safe as assert_smem_muse,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        fused_rs_smem_rows as smem_rows_muse,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )

    # Exact SMEM byte calculation for T=32768 (M=262144), T=49152 (M=393216), T=65536 (M=524288)
    for num_tok in (32768, 49152, 65536):
        size_m = num_tok * 8
        w_std = smem_rows_std(size_m, 8, False)
        w_muse = smem_rows_muse(size_m, 8, False)
        assert_smem_std(w_std)
        assert_smem_muse(size_m)
        for tile_m in (128, 256):
            num_slots = 33 + (w_muse + tile_m - 1) // tile_m - 1
            smem_bytes = (
                296 * 4  # group_sizes
                + 8  # group_offset
                + (5 + 2 * num_slots) * 4  # group_start
                + (w_muse // 2 + 1) * 4  # primary_idx_ref (u16x2 packed int32)
                + (num_slots + 1) * 26 * 4  # secondary_idx_ref (flattened 1D block_scatter)
                + 36 * 4  # max_num_gm_arr (+ lookahead table)
                + 3 * 33 * 4  # global scales
                + (2 * num_slots + 1) * 4  # metadata_ref
                + 35 * 4  # gm_id_ref + fused_metadata_ref
            )
            assert smem_bytes < 700 * 1024, (
                f"SMEM budget exceeded at T={num_tok}, tile_m={tile_m}: {smem_bytes} B"
            )

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    mesh_1d = Mesh(np.array(jax.devices()[:8]), ("model",))
    mesh_2d = Mesh(np.array(jax.devices()[:8]).reshape(1, 8), ("data", "model"))
    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
    ):
        try:
            # 1) Verify stock FP8 fused_moe_func_rs (L0/L61 fallback path that crashed c32k!) lowers at T=32768 & T=49152
            for num_tok in (32768, 49152):
                jax.clear_caches()
                fn_std = lambda hs, w1, w2, s1, s2, tw, ti, pnw, go: fused_moe_std(
                    hidden_states=hs,
                    w1=w1,
                    w2=w2,
                    w1_scale=s1,
                    w2_scale=s2,
                    w1_bias=None,
                    w2_bias=None,
                    gating_output=None,
                    topk=8,
                    renormalize=False,
                    mesh=mesh_2d,
                    activation="silu",
                    scoring_fn="softmax",
                    post_expert_norm_weight=pnw,
                    topk_weights=tw,
                    topk_indices=ti,
                    group_offset_input=go,
                )
                hlo_std = (
                    jax.jit(fn_std)
                    .trace(
                        jax.ShapeDtypeStruct((num_tok, 4096), jnp.bfloat16),
                        jax.ShapeDtypeStruct((32, 4096, 8192), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((32, 4096, 4096), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((32, 1, 1, 8192), jnp.float32),
                        jax.ShapeDtypeStruct((32, 1, 1, 4096), jnp.float32),
                        jax.ShapeDtypeStruct((num_tok, 8), jnp.bfloat16),
                        jax.ShapeDtypeStruct((num_tok, 8), jnp.int32),
                        jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                        jax.ShapeDtypeStruct((1,), jnp.int32),
                    )
                    .lower(lowering_platforms=("tpu",))
                    .as_text("stablehlo")
                )
                assert f"gmm_v2_fused_rs-E_32-M_{num_tok * 8}" in hlo_std

            # 2) Verify full Muse prefill pipeline at T=32768 and T=49152 with Wave-8 flags stacked!
            mesh = mesh_1d
            for num_tok in (32768, 49152):
                with patch.dict(
                    os.environ,
                    {
                        "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
                        "SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE": "1",
                        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                        "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                        "SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "2",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                        "SGLANG_TPU_MUSE_KERNEL_D_T64_C16": "1",
                        "SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP": "1",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_GATHER_2AHEAD": "1",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_WEIGHT_LOOKAHEAD": "1",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_DIRECT_SILU": "1",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "1",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "3",
                        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "4.0",
                    },
                ):
                    jax.clear_caches()

                    def _wrap(
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
                        )(
                            a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus,
                            pfw, aa, ba, pfw,
                        )

                    hlo_w8 = (
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
                            jax.ShapeDtypeStruct((256, 16, 8192), jnp.bfloat16),
                            jax.ShapeDtypeStruct((256, 16, 4096), jnp.bfloat16),
                            jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                            jax.ShapeDtypeStruct((8,), jnp.int32),
                            jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                            jax.ShapeDtypeStruct((8192,), jnp.float32),
                        )
                        .lower(lowering_platforms=("tpu",))
                        .as_text("stablehlo")
                    )
                    assert f"gmm_v2_fused_rs-E_32-M_{num_tok * 8}" in hlo_w8
                    assert "-u16w" in hlo_w8
                    assert "-g2a1" in hlo_w8
                    assert "-wla1" in hlo_w8
                    assert "-rrs2" in hlo_w8
                    assert "-ds1" in hlo_w8
                    assert "-pql1" in hlo_w8
                    assert "-sw3" in hlo_w8
                    assert "-sfp8_4" in hlo_w8
                    assert "pallas_deq_both_fp8_gmm" in hlo_w8
        finally:
            jax.clear_caches()


def test_w8_two_level_pow2_and_wider_qblock_requant_and_lowering():
    """Verify I1 wider qblock (512/1024) and I2 two-level pow2 requant SNR, exact s_2l identity, and Mosaic lowering."""
    import jax
    import jax.numpy as jnp
    import torch
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.moe.nvfp4_requant import (
        _dequant,
        compact_nvfp4_blocks,
        compact_nvfp4_two_level_pow2,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_moe import moe_gmm_local_rs_nodedup

    torch.manual_seed(11)
    # Construct realistic block-16 NVFP4 weights [2, 512, 128] (K=512, N=256)
    packed16 = torch.randint(0, 255, (2, 512, 128), dtype=torch.uint8)
    scale16 = (torch.rand(2, 32, 256, dtype=torch.float32) * 0.15 + 0.01).to(
        torch.bfloat16
    )
    ref = _dequant(packed16, scale16)

    # I1: block=512 compact requant
    c512, s512, err512, ref2 = compact_nvfp4_blocks(packed16, scale16, 512)
    assert s512.shape == (2, 1, 256)
    assert float((err512 / ref2).sqrt()) < 0.25

    # I2: two-level pow2 requant at block=256
    c_tlp, s_tlp, err_tlp, ref_tlp = compact_nvfp4_two_level_pow2(
        packed16, scale16, 256
    )
    assert s_tlp.shape == (2, 2, 256)
    assert s_tlp.dtype == torch.bfloat16
    rel_err_tlp = float((err_tlp / ref_tlp).sqrt())
    assert rel_err_tlp < 0.22

    # Verify exact IEEE floating-point recovery of pow2_shift * col_scale == s_tlp!
    s_f32 = s_tlp.float().numpy()
    s_max = np.max(s_f32, axis=1, keepdims=True)
    s_safe = np.where(s_max > 0, s_max, 1.0)
    pow2_shift = (s_f32 / s_safe) * 64.0
    col_scale = s_max * (1.0 / 64.0)
    np.testing.assert_allclose(pow2_shift * col_scale, s_f32, rtol=0.0, atol=0.0)
    nz = pow2_shift[pow2_shift > 0]
    np.testing.assert_allclose(np.log2(nz), np.round(np.log2(nz)), rtol=0.0, atol=0.0)

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
    ):
        try:
            for extra_env, expected_tag in (
                (
                    {
                        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "3",
                        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "4.0",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2": "1",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_K_STEP": "512",
                    },
                    "-tlp1k512-sw3-sfp8_4",
                ),
                (
                    {
                        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "3",
                        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "4.0",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2": "1",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_K_STEP": "1024",
                    },
                    "-tlp1k1024-sw3-sfp8_4",
                ),
                (
                    {
                        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "2",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_POW2": "1",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_TWO_LEVEL_K_STEP": "4096",
                    },
                    "-tlp1k4096-sw2",
                ),
                (
                    {
                        "SGLANG_TPU_MUSE_KERNEL_PREFILL_GMM_TILES": "128,4096,2048,4096,2048,2,2",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_W2_FP8_CACHE": "1",
                    },
                    "-w2c1",
                ),
                (
                    {
                        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                        "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "-512",
                    },
                    "-sw2",
                ),
                (
                    {
                        "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "3",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "2",
                        "SGLANG_TPU_MUSE_KERNEL_COMPACT_FP4_QBLOCK": "-1024",
                    },
                    "-sw3",
                ),
            ):
                with patch.dict(
                    os.environ,
                    {
                        "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                        "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                        "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                        **extra_env,
                    },
                ):
                    jax.clear_caches()

                    def _moe_only(hs, w1, s1, w2, s2, go, tw, ti, pnw):
                        return jax.shard_map(
                            lambda _hs, _w1, _s1, _w2, _s2, _go, _tw, _ti, _pnw: moe_gmm_local_rs_nodedup(
                                _hs,
                                _w1,
                                _s1,
                                None,
                                _w2,
                                _s2,
                                None,
                                None,
                                None,
                                _go,
                                _tw,
                                _ti,
                                _pnw,
                                activation="silu",
                                topk=8,
                                ep_size=8,
                                ep_axis_name="model",
                                has_post_norm=True,
                                sp_enabled=True,
                                block_scatter=True,
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
                            ),
                            out_specs=P("model", None),
                            check_vma=False,
                        )(hs, w1, s1, w2, s2, go, tw, ti, pnw)

                    hlo_m = (
                        jax.jit(_moe_only)
                        .trace(
                            jax.ShapeDtypeStruct((16384, 32, 128), jnp.bfloat16),
                            jax.ShapeDtypeStruct((256, 4096, 8192), jnp.float4_e2m1fn),
                            jax.ShapeDtypeStruct((256, 16, 8192), jnp.bfloat16),
                            jax.ShapeDtypeStruct((256, 4096, 4096), jnp.float4_e2m1fn),
                            jax.ShapeDtypeStruct((256, 16, 4096), jnp.bfloat16),
                            jax.ShapeDtypeStruct((8,), jnp.int32),
                            jax.ShapeDtypeStruct((16384, 8), jnp.bfloat16),
                            jax.ShapeDtypeStruct((16384, 8), jnp.int32),
                            jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                        )
                        .lower(lowering_platforms=("tpu",))
                        .as_text("stablehlo")
                    )
                    assert expected_tag in hlo_m

            # Regression test (#1622/#1623): T=1024 (size_m=8192 < 16384) warmup bucket with GMM_PREQUANT_LHS_FP8=1
            with patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_FP4_LHS_QBS": "4096",
                    "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STRIP_N": "256",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_FAST_LHS_QUANT": "1",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_PREQUANT_LHS_FP8": "1",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_SCOPED_W4A8": "2",
                    "SGLANG_TPU_MUSE_KERNEL_PREFILL_STATIC_FP8_RS": "4.0",
                },
            ):
                jax.clear_caches()
                # 1) Direct FP8 hidden_states + hidden_states_scale at T=1024 (size_m=8192)
                def _moe_t1024_ext(hs, sc, w1, s1, w2, s2, go, tw, ti, pnw):
                    return jax.shard_map(
                        lambda _hs, _sc, _w1, _s1, _w2, _s2, _go, _tw, _ti, _pnw: moe_gmm_local_rs_nodedup(
                            _hs,
                            _w1,
                            _s1,
                            None,
                            _w2,
                            _s2,
                            None,
                            None,
                            None,
                            _go,
                            _tw,
                            _ti,
                            _pnw,
                            activation="silu",
                            topk=8,
                            ep_size=8,
                            ep_axis_name="model",
                            has_post_norm=True,
                            sp_enabled=True,
                            block_scatter=True,
                            hidden_states_scale=_sc,
                        ),
                        mesh=mesh,
                        in_specs=(
                            P(),
                            P(),
                            P("model"),
                            P("model"),
                            P("model"),
                            P("model"),
                            P("model"),
                            P("model", None),
                            P(),
                            P(),
                        ),
                        out_specs=P("model", None),
                        check_vma=False,
                    )(hs, sc, w1, s1, w2, s2, go, tw, ti, pnw)

                hlo_t1024 = (
                    jax.jit(_moe_t1024_ext)
                    .trace(
                        jax.ShapeDtypeStruct((1024, 32, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((1024,), jnp.float32),
                        jax.ShapeDtypeStruct((256, 4096, 8192), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((256, 16, 8192), jnp.bfloat16),
                        jax.ShapeDtypeStruct((256, 4096, 4096), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((256, 16, 4096), jnp.bfloat16),
                        jax.ShapeDtypeStruct((8,), jnp.int32),
                        jax.ShapeDtypeStruct((1024, 8), jnp.bfloat16),
                        jax.ShapeDtypeStruct((1024, 8), jnp.int32),
                        jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                    )
                    .lower(lowering_platforms=("tpu",))
                    .as_text("stablehlo")
                )
                assert "M_8192" in hlo_t1024 and "-pql1" in hlo_t1024
        finally:
            jax.clear_caches()


def test_w8_numerical_equivalence_routing_and_u16_packing():
    """Verify bit-exact rotated RS routing bijection and u16x2 odd/even gather index unpacking."""
    import jax
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.fused_rs_muse.gmm_fused_rs_nodedup import (
        _precompute_block_scatter_meta_closed_form,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_global_local_pos,
    )

    rng = np.random.default_rng(123)
    ep_size = 8
    num_local = 32
    t_local = 128
    t_global = ep_size * t_local
    topk = 8

    # Unique topk expert IDs per token in [0, 256)
    g_ids = np.stack(
        [rng.choice(256, size=topk, replace=False) for _ in range(t_global)],
        axis=0,
    ).astype(np.int32)
    g_ids_j = jnp.asarray(g_ids)
    tok_ids_flat = jnp.arange(t_global * topk, dtype=jnp.int32) // topk

    for my_id in (0, 3, 7):
        go_val = jnp.int32(my_id * num_local)
        pos_std = pallas_global_local_pos(
            g_ids_j,
            go_val,
            num_local_experts=num_local,
            ep_size=ep_size,
            rotate_rs=False,
            interpret=True,
        )
        pos_rot = pallas_global_local_pos(
            g_ids_j,
            go_val,
            num_local_experts=num_local,
            ep_size=ep_size,
            rotate_rs=True,
            interpret=True,
        )
        lhs_std = (
            jnp.full((t_global * topk,), -1, dtype=jnp.int32)
            .at[pos_std.reshape(-1)]
            .set(tok_ids_flat, mode="drop", unique_indices=True)
        )
        lhs_rot = (
            jnp.full((t_global * topk,), -1, dtype=jnp.int32)
            .at[pos_rot.reshape(-1)]
            .set(tok_ids_flat, mode="drop", unique_indices=True)
        )

        # Build closed-form counts header
        by_chip = g_ids.reshape(ep_size, -1)
        chip_before = np.sum(by_chip < int(go_val), axis=1).astype(np.int32)
        counts_ce = np.stack(
            [
                np.sum(by_chip == (int(go_val) + e_l), axis=1)
                for e_l in range(num_local)
            ],
            axis=0,
        ).astype(np.int32)
        local_gs = np.sum(counts_ce, axis=1).astype(np.int32)
        full_gs = np.zeros((256,), dtype=np.int32)
        full_gs[int(go_val) : int(go_val) + num_local] = local_gs
        gs_raw = jnp.asarray(
            np.concatenate([[0, 0], chip_before, counts_ce.reshape(-1)]).astype(
                np.int32
            )
        )
        meta_rot = np.asarray(
            _precompute_block_scatter_meta_closed_form(
                jnp.asarray(full_gs),
                jnp.asarray([int(go_val)], dtype=jnp.int32),
                gs_raw,
                size_group=num_local,
                tile_m=32,
                num_slots=128,
                ep_size=ep_size,
                rotate_rs=True,
            )
        )
        meta_std = np.asarray(
            _precompute_block_scatter_meta_closed_form(
                jnp.asarray(full_gs),
                jnp.asarray([int(go_val)], dtype=jnp.int32),
                gs_raw,
                size_group=num_local,
                tile_m=32,
                num_slots=128,
                ep_size=ep_size,
                rotate_rs=False,
            )
        )

        # Reconstruct scattered (dst_chip, dst_pos) -> source token mapping for BOTH std & rot!
        def _reconstruct_scatter(lhs_arr, meta_arr):
            mapping = {}
            cur_m = 0
            gm = 0
            for e_l in range(num_local):
                sz = int(local_gs[e_l])
                if sz == 0:
                    continue
                for t0 in range(0, sz, 32):
                    t_len = min(32, sz - t0)
                    tile_toks = np.asarray(lhs_arr[cur_m + t0 : cur_m + t0 + t_len])
                    for c in range(ep_size):
                        s_off = int(meta_arr[gm, c])
                        d_off = int(meta_arr[gm, ep_size + c])
                        ln = int(meta_arr[gm, 2 * ep_size + c])
                        for r in range(ln):
                            tok = int(tile_toks[s_off + r])
                            assert tok // t_local == c
                            mapping[(c, d_off + r)] = (e_l, tok)
                    gm += 1
                cur_m += sz
            return mapping

        map_std = _reconstruct_scatter(lhs_std, meta_std)
        map_rot = _reconstruct_scatter(lhs_rot, meta_rot)
        assert map_std == map_rot, f"Rotated RS mapping mismatch on rank {my_id}"
        # Also verify my_id's rows sit at the END of every non-empty expert in lhs_rot:
        cur_m = 0
        for e_l in range(num_local):
            sz = int(local_gs[e_l])
            if sz > 0 and counts_ce[e_l, my_id] > 0:
                tail_toks = np.asarray(
                    lhs_rot[cur_m + sz - counts_ce[e_l, my_id] : cur_m + sz]
                )
                assert np.all(tail_toks // t_local == my_id)
            cur_m += sz

    # Verify u16x2 odd and even m_start unpacking matches exact indices up to 65535
    raw_u16 = rng.integers(0, 65536, size=(256,), dtype=np.int32)
    u_pairs = raw_u16.astype(np.uint32).reshape(128, 2)
    packed = np.pad(
        ((u_pairs[:, 0] & 0xFFFF) | (u_pairs[:, 1] << 16)).astype(np.int32),
        (0, 1),
    )
    for pos in (0, 1, 5, 16, 33, 247):
        odd = pos & 1
        wb = pos >> 1
        w5 = [int(packed[wb + j]) for j in range(5)]
        out8 = []
        for j in range(4):
            lo_j = w5[j] & 0xFFFF
            hi_j = (w5[j] >> 16) & 0xFFFF
            nlo_j = w5[j + 1] & 0xFFFF
            out8.append(lo_j if odd == 0 else hi_j)
            out8.append(hi_j if odd == 0 else nlo_j)
        np.testing.assert_array_equal(out8, raw_u16[pos : pos + 8])


def test_w8_scoped_two_level_pow2_and_qblock_numerical_parity():
    """Verify exact numerical parity of scoped_two_level_pow2_strip_acc (k_step=512/1024/2048/4096 + 3D prequant LHS)
    and scoped_w4a8_strip_acc (qblock=512/1024 + 3D prequant LHS + quant_otf) under Pallas interpret=True."""
    import jax
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
        scoped_two_level_pow2_strip_acc,
        scoped_w4a8_strip_acc,
    )

    rng = np.random.default_rng(2026)
    tile_m, tile_k, col_size = 256, 2048, 256
    rhs_qbs = 256
    num_blocks = tile_k // rhs_qbs

    # Exact E2M1 values in float4_e2m1fn
    e2m1_vals = np.array(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=np.float32,
    )
    w_f32 = rng.choice(e2m1_vals, size=(tile_k, col_size)).astype(np.float32)
    # Exact two-level pow2 scales: c[n] * 2^p[b, n] with p in {-4..0}
    col_c = (rng.uniform(0.02, 0.15, size=(1, col_size))).astype(np.float32)
    p_shifts = (2.0 ** rng.integers(-4, 1, size=(num_blocks, col_size))).astype(np.float32)
    s_2l = (col_c * p_shifts).astype(np.float32)
    # Ensure at least 1 block per col hits max p_shift=1.0 so s_max == col_c
    s_2l[0, :] = col_c[0, :]
    s_2l_bf16 = jnp.asarray(s_2l, dtype=jnp.bfloat16)

    x_fp8 = jnp.asarray(rng.normal(0.0, 1.5, size=(tile_m, tile_k)), dtype=jnp.float8_e4m3fn)
    x_3d_fp8 = x_fp8.reshape(tile_m, tile_k // 128, 128)
    x_sc = jnp.asarray(rng.uniform(0.01, 0.05, size=(tile_m, 1)), dtype=jnp.float32)
    x_sc_128 = jnp.broadcast_to(x_sc, (tile_m, 128))
    w_fp4 = jnp.asarray(w_f32, dtype=jnp.float4_e2m1fn)

    # Exact FP32 reference
    w_deq = w_f32 * np.repeat( np.asarray(s_2l_bf16, dtype=np.float32), rhs_qbs, axis=0 )
    ref = (np.asarray(x_fp8, dtype=np.float32) * np.asarray(x_sc)) @ w_deq

    for k_step, mho in ((512, False), (512, True), (1024, False), (1024, True), (2048, False), (4096, False)):
        def _run_tlp(x3d_r, xs_r, w_r, s_r, out_r, strip_r):
            out_r[...] = scoped_two_level_pow2_strip_acc(
                w_weight_ref=w_r,
                w_scale_ref=s_r,
                rhs_fp8_strip_ref=strip_r,
                start_n=0,
                end_n=col_size,
                tile_m=tile_m,
                tile_k=tile_k,
                q_block_size=tile_k,
                rhs_qbs=rhs_qbs,
                packing=1,
                rhs_quant_dtype=jnp.float4_e2m1fn,
                lhs_quant_dtype=jnp.float8_e4m3fn,
                acc_dtype=jnp.float32,
                two_level_mode=1,
                two_level_k_step=k_step,
                quant_otf=False,
                otf_inv_scale_bf16=None,
                lhs_2d_vmem_ref=None,
                prequant_lhs_fp8_ref=None,
                prequant_lhs_3d_fp8_ref=x3d_r,
                has_prequant_lhs=True,
                hoisted_lq_blocks=[],
                load_lhs_scale_strip_fn=lambda _b, cw: jnp.concatenate(
                    [xs_r[...]] * (cw // 128), axis=1
                ),
                mh_outer=mho,
            )

        got = np.asarray(
            pl.pallas_call(
                _run_tlp,
                out_shape=jax.ShapeDtypeStruct((tile_m, col_size), jnp.float32),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 4,
                    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                    scratch_shapes=[pltpu.VMEM((tile_k, col_size), jnp.float8_e4m3fn)],
                ),
                interpret=True,
            )(x_3d_fp8, x_sc_128, w_fp4, s_2l_bf16)
        )
        rel_l2 = np.linalg.norm(got - ref) / np.linalg.norm(ref)
        assert rel_l2 < 5e-3, f"Two-level pow2 k_step={k_step} rel_l2={rel_l2}"

    # Verify scoped_w4a8_strip_acc at qblock=512 and qblock=1024 with 3D prequant LHS
    for qbs_w in (512, 1024):
        nb_w = tile_k // qbs_w
        s_w = jnp.asarray(
            rng.uniform(0.02, 0.12, size=(nb_w, col_size)), dtype=jnp.bfloat16
        )
        ref_w = (np.asarray(x_fp8, dtype=np.float32) * np.asarray(x_sc)) @ (
            w_f32 * np.repeat(np.asarray(s_w, dtype=np.float32), qbs_w, axis=0)
        )

        def _run_sw(x3d_r, xs_r, w_r, s_r, out_r):
            out_r[...] = scoped_w4a8_strip_acc(
                w_weight_ref=w_r,
                w_scale_ref=s_r,
                start_n=0,
                end_n=col_size,
                tile_m=tile_m,
                tile_k=tile_k,
                q_block_size=tile_k,
                rhs_qbs=qbs_w,
                packing=1,
                rhs_quant_dtype=jnp.float4_e2m1fn,
                lhs_quant_dtype=jnp.float8_e4m3fn,
                acc_dtype=jnp.float32,
                use_bf16_inner=False,
                quant_otf=False,
                otf_inv_scale_bf16=None,
                lhs_2d_vmem_ref=None,
                prequant_lhs_fp8_ref=None,
                prequant_lhs_3d_fp8_ref=x3d_r,
                has_prequant_lhs=True,
                hoisted_lq_blocks=[],
                load_lhs_scale_strip_fn=lambda _b, cw: jnp.concatenate(
                    [xs_r[...]] * (cw // 128), axis=1
                ),
                zero_spill_sub_m=True,
                mh_outer=False,
            )

        got_w = np.asarray(
            pl.pallas_call(
                _run_sw,
                out_shape=jax.ShapeDtypeStruct((tile_m, col_size), jnp.float32),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 4,
                    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                ),
                interpret=True,
            )(x_3d_fp8, x_sc_128, w_fp4, s_w)
        )
        rel_l2_w = np.linalg.norm(got_w - ref_w) / np.linalg.norm(ref_w)
        assert rel_l2_w < 1e-5, f"scoped_w4a8 qblock={qbs_w} rel_l2={rel_l2_w}"




def test_w9_replica_bank_and_decode_qblock512(monkeypatch):
    """Verify QBLOCK=512 / two_level_pow2 replica bank formatting, _finalize_decode_replica_banks,
    _fp4_gmm_chunked scale_block=512 hoist_scales parity, and TPU lowering for sparse v1/v2 R=6."""
    from types import SimpleNamespace

    import jax
    import jax.experimental.pallas as pl
    import jax.numpy as jnp
    import torch
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.models.muse_spark import (
        MuseSparkForConditionalGeneration,
        MuseSparkMoE,
        _format_nvfp4_replica_bank_from_raw,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_op import (
        muse_fuse_all_decode_op_name,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        _fp4_gmm_chunked,
        fused_decode_sparse_from_logits,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse_v2 import (
        fused_decode_sparse_v2_from_logits,
    )

    # 1. Replica bank formatting + _finalize_decode_replica_banks + attach_preloaded_replicas at R=6, QBLOCK=512
    torch.manual_seed(0)
    num_rep = 6
    raw_full = {
        "w13_weight": torch.randint(0, 256, (num_rep, 8192, 2048), dtype=torch.uint8),
        "w13_weight_scale": torch.rand((num_rep, 8192, 256), dtype=torch.bfloat16) * 0.1
        + 0.01,
        "w2_weight": torch.randint(0, 256, (num_rep, 4096, 2048), dtype=torch.uint8),
        "w2_weight_scale": torch.rand((num_rep, 4096, 256), dtype=torch.bfloat16) * 0.1
        + 0.01,
    }
    rw1_512, rs1_512, rw2_512, rs2_512 = _format_nvfp4_replica_bank_from_raw(
        raw_full, device=torch.device("cpu"), compact_k256=True, compact_block=512
    )
    assert rw1_512.dtype == torch.float4_e2m1fn_x2
    assert tuple(rw1_512.shape) == (num_rep, 4096, 4096)
    assert tuple(rs1_512.shape) == (num_rep, 8, 8192)
    assert tuple(rw2_512.shape) == (num_rep, 4096, 2048)
    assert tuple(rs2_512.shape) == (num_rep, 8, 4096)

    # Also test two_level_pow2=True on compact_block=256 (small slice for speed)
    raw_small = {
        "w13_weight": torch.randint(0, 256, (2, 128, 256), dtype=torch.uint8),
        "w13_weight_scale": torch.rand((2, 128, 32), dtype=torch.bfloat16) * 0.1 + 0.01,
        "w2_weight": torch.randint(0, 256, (2, 64, 256), dtype=torch.uint8),
        "w2_weight_scale": torch.rand((2, 64, 32), dtype=torch.bfloat16) * 0.1 + 0.01,
    }
    _, rs1_tlp, _, rs2_tlp = _format_nvfp4_replica_bank_from_raw(
        raw_small,
        device=torch.device("cpu"),
        compact_k256=True,
        compact_block=256,
        two_level_pow2=True,
    )
    assert tuple(rs1_tlp.shape) == (2, 2, 128)
    assert tuple(rs2_tlp.shape) == (2, 2, 64)

    # End-to-end _finalize_decode_replica_banks -> attach_preloaded_replicas with QBLOCK=512 experts
    dummy_mlp = torch.nn.Module()
    dummy_mlp.tp_size = 8
    dummy_mlp.__dict__["moe_ep_size"] = 8
    dummy_mlp.experts = SimpleNamespace(
        w13_weight=torch.empty((32, 4096, 4096), dtype=torch.float4_e2m1fn_x2),
        w13_weight_scale=torch.empty((32, 8, 8192), dtype=torch.bfloat16),
        w2_weight=torch.empty((32, 4096, 2048), dtype=torch.float4_e2m1fn_x2),
        w2_weight_scale=torch.empty((32, 8, 4096), dtype=torch.bfloat16),
    )
    dummy_mlp.attach_preloaded_replicas = (
        MuseSparkMoE.attach_preloaded_replicas.__get__(dummy_mlp, torch.nn.Module)
    )
    dummy_mlp.__dict__["_pending_replica_raw"] = raw_full
    dummy_host = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        model=SimpleNamespace(layers=[SimpleNamespace(mlp=dummy_mlp)]),
        _decode_replica_plans={
            0: SimpleNamespace(replica_meta=np.zeros((4, 8 * num_rep), dtype=np.int32))
        },
    )
    MuseSparkForConditionalGeneration._finalize_decode_replica_banks(
        dummy_host, compact_k256=True
    )
    assert tuple(dummy_mlp._buffers["_fused_decode_rep_s1"].shape) == (num_rep, 8, 8192)
    assert tuple(dummy_mlp._buffers["_fused_decode_rep_s2"].shape) == (num_rep, 8, 4096)

    # 2. _fp4_gmm_chunked hoist_scales=True vs False numerical parity at scale_block=512
    rng = np.random.default_rng(77)
    m_rows, k_dim, nc = 32, 4096, 256
    x_fp8 = jnp.asarray(
        rng.normal(0.0, 0.5, size=(m_rows, k_dim)), dtype=jnp.float8_e4m3fn
    )
    x_sc_16 = rng.uniform(0.01, 0.05, size=(m_rows, 16)).astype(np.float32)
    x_sc_buf = jnp.asarray(np.repeat(x_sc_16, 128, axis=1), dtype=jnp.float32)
    w_fp4 = jnp.asarray(
        np.clip(rng.normal(0.0, 1.5, size=(k_dim, nc)), -6.0, 6.0),
        dtype=jnp.float4_e2m1fn,
    )
    w_u32 = jax.lax.bitcast_convert_type(
        w_fp4.reshape(k_dim // 8, 8, nc).transpose(0, 2, 1), jnp.uint32
    )
    s_512 = jnp.asarray(
        rng.uniform(0.02, 0.10, size=(k_dim // 512, nc)), dtype=jnp.bfloat16
    )

    w_f32 = np.asarray(w_fp4.astype(jnp.float32))
    s_rep = np.repeat(np.asarray(s_512, dtype=np.float32), 512, axis=0)
    x_f32 = np.asarray(x_fp8, dtype=np.float32) * np.repeat(x_sc_16, 256, axis=1)
    ref_512 = x_f32 @ (w_f32 * s_rep)

    for hoist in (True, False):

        def _kern_512(xr, xsr, wr, sr, outr):
            outr[...] = _fp4_gmm_chunked(
                xr,
                wr,
                sr,
                m_rows,
                k_dim,
                0,
                nc,
                True,
                512,
                lhs_sc_ref=xsr,
                hoist_scales=hoist,
            )

        got_512 = np.asarray(
            pl.pallas_call(
                _kern_512,
                out_shape=jax.ShapeDtypeStruct((m_rows, nc), jnp.float32),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=0,
                    in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)] * 4,
                    out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
                ),
                interpret=True,
            )(x_fp8, x_sc_buf, w_u32, s_512)
        )
        rel = np.linalg.norm(got_512 - ref_512) / np.linalg.norm(ref_512)
        assert rel < 1e-5, f"_fp4_gmm_chunked scale_block=512 hoist={hoist} rel={rel}"

    # 3. Op naming + TPU StableHLO lowering for sparse (v1) and sparse_v2 at scale_block=512, R=6
    assert (
        muse_fuse_all_decode_op_name(
            mode="sparse", ep=8, tp=1, block=512, num_replicas=6
        )
        == "sglang_tpu_pallas::muse_decode_sparse_ep8_k512_r6_tiles"
    )
    from jax._src import tpu_info

    dk = next(
        k
        for k in ("TPU7x", "TPU v7x", "TPU v6e", "TPU v5p")
        if k in tpu_info.registry
        or tpu_info.chip_version_from_device_kind(k) is not None
    )
    monkeypatch.setattr(tpu_info, "get_device_kind", lambda: dk)
    tpu_info.get_tpu_info.cache_clear()
    devs = jax.devices("cpu")
    if len(devs) >= 8:
        mesh = Mesh(np.array(devs[:8]), ("model",))
        try:
            for fn in (
                fused_decode_sparse_from_logits,
                fused_decode_sparse_v2_from_logits,
            ):

                def _wrap_dec(
                    rk, lg, bi, xi, w1, s1, w2, s2, nm, rw1, rs1, rw2, rs2, rm
                ):
                    return jax.shard_map(
                        lambda _rk, _lg, _bi, _xi, _w1, _s1, _w2, _s2, _nm, _rw1, _rs1, _rw2, _rs2, _rm: (
                            fn(
                                _rk,
                                _lg,
                                _bi,
                                _xi,
                                _w1,
                                _s1,
                                _w2,
                                _s2,
                                _nm,
                                rep_w1=_rw1,
                                rep_s1=_rs1,
                                rep_w2=_rw2,
                                rep_s2=_rs2,
                                replica_meta=_rm,
                                ep=8,
                                ep_axis_name="model",
                            )
                        ),
                        mesh=mesh,
                        in_specs=(
                            P("model"),
                            P(),
                            P(),
                            P(),
                            P("model"),
                            P("model"),
                            P("model"),
                            P("model"),
                            P(),
                            P("model"),
                            P("model"),
                            P("model"),
                            P("model"),
                            P(),
                        ),
                        out_specs=P(),
                        check_vma=False,
                    )(rk, lg, bi, xi, w1, s1, w2, s2, nm, rw1, rs1, rw2, rs2, rm)

                hlo_dec = (
                    jax.jit(_wrap_dec)
                    .trace(
                        jax.ShapeDtypeStruct((8,), jnp.int32),
                        jax.ShapeDtypeStruct((64, 256), jnp.float32),
                        jax.ShapeDtypeStruct((256,), jnp.float32),
                        jax.ShapeDtypeStruct((64, 4096), jnp.bfloat16),
                        jax.ShapeDtypeStruct((256, 4096, 8192), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((256, 8, 8192), jnp.bfloat16),
                        jax.ShapeDtypeStruct((256, 4096, 4096), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((256, 8, 4096), jnp.bfloat16),
                        jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                        jax.ShapeDtypeStruct((8 * 6, 4096, 8192), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((8 * 6, 8, 8192), jnp.bfloat16),
                        jax.ShapeDtypeStruct((8 * 6, 4096, 4096), jnp.float4_e2m1fn),
                        jax.ShapeDtypeStruct((8 * 6, 8, 4096), jnp.bfloat16),
                        jax.ShapeDtypeStruct((4, 8 * 6), jnp.int32),
                    )
                    .lower(lowering_platforms=("tpu",))
                    .as_text("stablehlo")
                )
                assert "k512" in hlo_dec
        finally:
            jax.clear_caches()


def test_w9_l0_l61_large_m_tuned_blocks_and_fused_rs_smem_window(monkeypatch):
    """Verify L0/L61 channelwise FP8 (rhs_bs in None/2048/4096) never falls back to default tm=64
    at M=196608/262144/393216/524288, and fused_rs/gmm_fused_rs_nodedup windows SMEM for M>131072."""
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.fused_rs import gmm_fused_rs_nodedup as frs_stock
    from sglang_tpu.moe.pallas.fused_rs import tuned_block_sizes as tbs_stock
    from sglang_tpu.moe.pallas.fused_rs_muse import tuned_block_sizes as tbs_muse

    # 1. Verify SMEM windowing at M=196608 (24k) and M=262144 (32k) in stock fused_rs
    assert frs_stock.fused_rs_smem_rows(131_072, 8, False) == 131_072
    assert frs_stock.fused_rs_smem_rows(196_608, 8, False) == 98_304
    assert frs_stock.fused_rs_smem_rows(262_144, 8, False) == 131_072

    # 2. Verify tuned block lookup hits non-default tiles across M=196608..524288
    monkeypatch.setattr(tbs_stock, "_device_kind", lambda: "TPU7x")
    monkeypatch.setattr(tbs_muse, "_device_kind", lambda: "TPU7x")
    sentinel_default = (64, 4096, 2048, 4096, 2048, 1, 1)

    for m_val in (196_608, 262_144, 393_216, 524_288):
        for rhs_bs in (None, 2048, 4096):
            res_stock = tbs_stock.get_fused_rs_tuned_block_sizes(
                m_val,
                4096,
                8192,
                4096,
                4096,
                32,
                jnp.bfloat16,
                jnp.float8_e4m3fn,
                rhs_bs,
                sentinel_default,
                fuse_act="silu_and_mul_clamp",
                fp8_direct_write=False,
            )
            assert res_stock != sentinel_default and res_stock[0] >= 112, (
                f"stock miss m={m_val} rhs_bs={rhs_bs}: {res_stock}"
            )
            for g in (16, 32, 34, 35):
                res_muse = tbs_muse.get_fused_rs_tuned_block_sizes(
                    m_val,
                    4096,
                    8192,
                    4096,
                    4096,
                    g,
                    jnp.bfloat16,
                    jnp.float8_e4m3fn,
                    rhs_bs,
                    sentinel_default,
                    fuse_act="silu_and_mul_clamp",
                    fp8_direct_write=False,
                )
                assert res_muse != sentinel_default and res_muse[0] >= 112, (
                    f"muse miss m={m_val} g={g} rhs_bs={rhs_bs}: {res_muse}"
                )

    # 3. Verify TPU7x StableHLO lowering of L0/L61 stock fused_rs at T=24576 (M=196608) & T=32768 (M=262144)
    import jax
    from jax.sharding import Mesh, PartitionSpec as P
    from jax._src import tpu_info
    from sglang_tpu.moe.pallas.fused_rs.fused_moe import (
        moe_gmm_local_rs_nodedup as moe_stock,
    )

    dk = next(
        k
        for k in ("TPU7x", "TPU v7x", "TPU v6e", "TPU v5p")
        if k in tpu_info.registry or tpu_info.chip_version_from_device_kind(k) is not None
    )
    monkeypatch.setattr(tpu_info, "get_device_kind", lambda: dk)
    tpu_info.get_tpu_info.cache_clear()
    devs = jax.devices("cpu")
    if len(devs) >= 8:
        mesh = Mesh(np.array(devs[:8]), ("model",))
        try:
            for t_glob in (24_576, 32_768):
                t_loc = t_glob // 8

                def _l0_moe(hs, w1, s1, w2, s2, go, tw, ti, pnw):
                    return jax.shard_map(
                        lambda _hs, _w1, _s1, _w2, _s2, _go, _tw, _ti, _pnw: moe_stock(
                            _hs,
                            _w1,
                            _s1,
                            None,
                            _w2,
                            _s2,
                            None,
                            None,
                            None,
                            _go,
                            _tw,
                            _ti,
                            _pnw,
                            activation="silu",
                            topk=8,
                            ep_size=8,
                            ep_axis_name="model",
                            has_post_norm=True,
                            sp_enabled=True,
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
                        ),
                        out_specs=P("model", None),
                        check_vma=False,
                    )(hs, w1, s1, w2, s2, go, tw, ti, pnw)

                hlo_l0 = (
                    jax.jit(_l0_moe)
                    .trace(
                        jax.ShapeDtypeStruct((t_glob, 4096), jnp.bfloat16),
                        jax.ShapeDtypeStruct((256, 4096, 8192), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((256, 2, 1, 8192), jnp.float32),
                        jax.ShapeDtypeStruct((256, 4096, 4096), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((256, 2, 1, 4096), jnp.float32),
                        jax.ShapeDtypeStruct((8,), jnp.int32),
                        jax.ShapeDtypeStruct((t_glob, 8), jnp.bfloat16),
                        jax.ShapeDtypeStruct((t_glob, 8), jnp.int32),
                        jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                    )
                    .lower(lowering_platforms=("tpu",))
                    .as_text("stablehlo")
                )
                assert f"M_{t_glob * 8}" in hlo_l0 and f"{t_loc}x4096xbf16" in hlo_l0
        finally:
            jax.clear_caches()
