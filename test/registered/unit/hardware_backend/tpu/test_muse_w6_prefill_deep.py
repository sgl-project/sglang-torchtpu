# SPDX-License-Identifier: Apache-2.0
"""Unit & Mosaic TPU7x HLO verification tests for Wave-6 Prefill Stage 1+2 Fused B1+B2."""

from __future__ import annotations

import ast
import os
import pathlib
from unittest.mock import patch

import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pallas_export import tpu_info_for_cpu_export


def test_w6_fused_b1_b2_default_off():
    """Wave-6 SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2 must default to 0 (strictly off)."""
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2.get() == 0


def test_no_bare_envfield_bool_evaluation_on_w6_files():
    """Ensure all modified Wave-6 files call .get() on EnvField rather than bare truthiness."""
    repo = pathlib.Path(__file__).resolve().parents[5]
    rel_paths = (
        "sglang_tpu/muse_prefill_kernels/integration.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/kernel_b2.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/pipeline.py",
    )
    checked_flags = {
        "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2",
        "SGLANG_TPU_MUSE_KERNEL_FP4_ACC_MODE",
        "SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE",
        "SGLANG_TPU_MUSE_KERNEL_DEQUANT_1PASS",
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


def test_pallas_fused_b1_b2_all_modes_cpu_export_hlo_zero_partition_id():
    """AOT-lower pallas_fused_b1_b2_wpool_quant (full & half, bf16 & fp8) + dequant to TPU7x StableHLO."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_b2 import (
        dequant_both_halves_gmm_layout,
        dequant_single_gmm_layout,
        pallas_fused_b1_b2_wpool_quant,
    )

    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
    ):
        for fp8_wp in (False, True):
            for half_idx in (None, 0, 1):
                fn = jax.jit(
                    lambda a, s, aa, ba, pfw, wr, eb, wp, wps, pew, fp8=fp8_wp, hi=half_idx: pallas_fused_b1_b2_wpool_quant(
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
                        router_mode="bf16_hilo",
                        fp8_wpool=fp8,
                        half_idx=hi,
                    )
                )
                lowered = fn.trace(
                    jax.ShapeDtypeStruct((1024, 8192), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1024, 8192), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8192, 256), jnp.float32),
                    jax.ShapeDtypeStruct((256,), jnp.float32),
                    jax.ShapeDtypeStruct((8192, 4096), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((4096,), jnp.float32),
                    jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                ).lower(lowering_platforms=("tpu",))
                hlo = lowered.as_text("stablehlo")
                assert "custom_call" in hlo
                assert "partition_id" not in hlo

        deq_1p = jax.jit(dequant_single_gmm_layout).trace(
            jax.ShapeDtypeStruct((8, 1024, 4224), jnp.int8)
        ).lower(lowering_platforms=("tpu",))
        assert "partition_id" not in deq_1p.as_text("stablehlo")

        deq_2h = jax.jit(
            lambda a0, a1: dequant_both_halves_gmm_layout(a0, None, a1, None)
        ).trace(
            jax.ShapeDtypeStruct((8, 512, 4224), jnp.int8),
            jax.ShapeDtypeStruct((8, 512, 4224), jnp.int8),
        ).lower(lowering_platforms=("tpu",))
        assert "partition_id" not in deq_2h.as_text("stablehlo")


def test_pipeline_fused_b1_b2_modes_1_to_4_shard_map_export():
    """Verify full shard_map MoE pipeline lowers cleanly across FUSED_B1_B2=1..4 on 8-device mesh."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    import numpy as np

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    with (
        patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
        tpu_info_for_cpu_export(),
    ):
        for mode in ("1", "2", "3", "4"):
            with patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": mode,
                    "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
                    "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                    "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                },
            ):
                fn = jax.jit(
                    lambda a, s, aa, ba, pfw, wr, eb, wp, wps, pew, w1, w2, s1, s2, pnw, go, wu, wus: fused_moe_full_bridge_and_ffn(
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
                        pfw,
                        aa,
                        ba,
                        pfw,
                        mesh=mesh,
                        router_mode="bf16_hilo",
                        b2_layout="nstream",
                        b2_two_half=True,
                        b2_deq0_first=True,
                        hoist_routing=True,
                        d_layout="4slab",
                        d_tok_tile=64,
                        b2_deq0_new_buffer=True,
                    )
                )
                lowered = fn.trace(
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
                assert "pallas_fused_b1_b2" in hlo
                assert "partition_id" not in hlo
