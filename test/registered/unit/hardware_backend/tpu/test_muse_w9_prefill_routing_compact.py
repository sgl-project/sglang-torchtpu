# SPDX-License-Identifier: Apache-2.0
"""Unit, TPU7x Mosaic lowering, and skewed numerical parity tests for Wave-10 Prefill Routing."""

from __future__ import annotations

import ast
import contextlib
import os
import pathlib
from unittest import mock

import numpy as np
import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

_NEW_W10_FLAGS = (
    "SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK",
    "SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT",
)


@contextlib.contextmanager
def _tpu7x_cpu_context(*, interpret: bool = False):
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    real_pallas_call = pl.pallas_call

    def _wrapped_pallas_call(*args, **kwargs):
        if interpret:
            kwargs["interpret"] = pltpu.InterpretParams()
        return real_pallas_call(*args, **kwargs)

    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
            return_value="TPU v7",
        ),
        tpu_info_for_cpu_export(),
        mock.patch.object(
            pl, "pallas_call", _wrapped_pallas_call if interpret else real_pallas_call
        ),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        yield


def test_w10_routing_flags_default_off_and_dot_get_hygiene():
    """Verify new flags default strictly OFF, use .get() everywhere, and key _w8_carb_cache_tag."""
    from sglang_tpu.muse_prefill_kernels.integration import _w8_carb_cache_tag

    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT.get() == 0

    base_tag = _w8_carb_cache_tag()
    with mock.patch.dict(
        os.environ, {"SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK": "1"}
    ):
        tag_rnp = _w8_carb_cache_tag()
    with mock.patch.dict(
        os.environ, {"SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT": "1"}
    ):
        tag_cls1 = _w8_carb_cache_tag()
    with mock.patch.dict(
        os.environ, {"SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT": "2"}
    ):
        tag_cls2 = _w8_carb_cache_tag()
    assert len({base_tag, tag_rnp, tag_cls1, tag_cls2}) == 4

    repo = pathlib.Path(__file__).resolve().parents[5]
    checked = set(_NEW_W10_FLAGS)
    for rel in (
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/routing_pallas.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/pipeline.py",
        "sglang_tpu/muse_prefill_kernels/integration.py",
    ):
        tree = ast.parse((repo / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in checked:
                has_get = any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "get"
                    and c.func.value is node
                    for c in ast.walk(tree)
                )
                assert has_get, f"{rel}:{node.lineno} accesses {node.attr} without .get()"


@pytest.mark.parametrize("t_local", [2048, 3072])
@pytest.mark.parametrize("has_pad_sentinel", [False, True])
def test_routing_native_pack_tpu7x_lowering_and_exact_parity(
    t_local: int, has_pad_sentinel: bool
):
    """Verify (8, 128)-native packing eliminates stablehlo.pad and matches stock bit-for-bit."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax import export

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_local_inv_perm_and_counts,
    )

    ep_size = 8
    num_local_experts = 34
    t_global = ep_size * t_local
    total_buckets = ep_size * num_local_experts + (1 if has_pad_sentinel else 0)

    with _tpu7x_cpu_context(interpret=False):
        stock_exp = export.export(
            jax.jit(
                lambda l, g, go, me: pallas_local_inv_perm_and_counts(
                    l,
                    g,
                    go,
                    me,
                    ep_size=ep_size,
                    num_local_experts=num_local_experts,
                    total_id_buckets=total_buckets,
                    has_pad_sentinel=has_pad_sentinel,
                    native_pack=False,
                )
            ),
            platforms=("tpu",),
        )(
            jax.ShapeDtypeStruct((t_local, 8), jnp.int32),
            jax.ShapeDtypeStruct((t_global, 8), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
        )
        assert "stablehlo.pad" in stock_exp.mlir_module()

        with mock.patch.dict(
            os.environ, {"SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK": "1"}
        ):
            nat_exp = export.export(
                jax.jit(
                    lambda l, g, go, me: pallas_local_inv_perm_and_counts(
                        l,
                        g,
                        go,
                        me,
                        ep_size=ep_size,
                        num_local_experts=num_local_experts,
                        total_id_buckets=total_buckets,
                        has_pad_sentinel=has_pad_sentinel,
                    )
                ),
                platforms=("tpu",),
            )(
                jax.ShapeDtypeStruct((t_local, 8), jnp.int32),
                jax.ShapeDtypeStruct((t_global, 8), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
                jax.ShapeDtypeStruct((), jnp.int32),
            )
        nat_mlir = nat_exp.mlir_module()
        assert "stablehlo.pad" not in nat_mlir
        assert f"pallas_moe_sortfree_route_np_tl{t_local}" in nat_mlir

    rng = np.random.default_rng(1000 + t_local + int(has_pad_sentinel))
    raw_g = rng.integers(0, ep_size * num_local_experts, size=(t_global, 8), dtype=np.int32)
    if has_pad_sentinel:
        raw_g[-256:, :] = ep_size * num_local_experts
    my_rank = 3
    g_ids = jnp.asarray(raw_g)
    l_ids = g_ids[my_rank * t_local : (my_rank + 1) * t_local]
    go_val = jnp.int32(my_rank * num_local_experts)
    me_val = jnp.int32(my_rank)

    with _tpu7x_cpu_context(interpret=True):
        ref_outs = pallas_local_inv_perm_and_counts(
            l_ids,
            g_ids,
            go_val,
            me_val,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            total_id_buckets=total_buckets,
            has_pad_sentinel=has_pad_sentinel,
            native_pack=False,
        )
        with mock.patch.dict(
            os.environ, {"SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK": "1"}
        ):
            got_outs = pallas_local_inv_perm_and_counts(
                l_ids,
                g_ids,
                go_val,
                me_val,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
                total_id_buckets=total_buckets,
                has_pad_sentinel=has_pad_sentinel,
            )

    for idx, (r, g) in enumerate(zip(ref_outs, got_outs)):
        assert np.array_equal(
            np.asarray(r), np.asarray(g)
        ), f"Output {idx} diverged for t_local={t_local}, sentinel={has_pad_sentinel}"


@pytest.mark.parametrize("t_global", [8192, 16384, 24576, 32768])
@pytest.mark.parametrize("compact_mode", [1, 2])
@pytest.mark.parametrize("rotate_rs", [False, True])
def test_compact_local_sort_tpu7x_lowering_zero_scatter(
    t_global: int, compact_mode: int, rotate_rs: bool
):
    """Verify COMPACT_LOCAL_SORT={1,2} lowers cleanly on TPU7x with zero scatter ops."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax import export

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        compute_block_scatter_routing,
    )

    ep_size = 8
    num_local_experts = 34
    hdr_len = 2 + ep_size + num_local_experts * ep_size
    rot_env = "2" if rotate_rs else "0"

    with (
        _tpu7x_cpu_context(interpret=False),
        mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": rot_env,
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT": str(compact_mode),
            },
        ),
    ):
        exp = export.export(
            jax.jit(
                lambda g, go, me, gs, hdr, tr: compute_block_scatter_routing(
                    g,
                    go,
                    me,
                    ep_size=ep_size,
                    num_local_experts=num_local_experts,
                    topk=8,
                    precomputed_counts=(gs, hdr, tr),
                )
            ),
            platforms=("tpu",),
        )(
            jax.ShapeDtypeStruct((t_global, 8), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((ep_size * num_local_experts,), jnp.int32),
            jax.ShapeDtypeStruct((hdr_len,), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
        )
        mlir = exp.mlir_module()
        assert "stablehlo.scatter" not in mlir
        if compact_mode == 1:
            assert f"pallas_compact_local_keys_tg{t_global}_el{num_local_experts}" in mlir


@pytest.mark.parametrize("t_global", [16384, 24576])
@pytest.mark.parametrize("rotate_rs", [False, True])
def test_compact_local_sort_all_tiers_and_100pct_skew_parity(
    t_global: int, rotate_rs: bool
):
    """Verify bit-exact lhs_indices parity across Tier 0 (32k), Tier 1 (64k), burst skew, and 100% skew."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        compute_block_scatter_routing,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_compact_local_packed_keys,
    )

    ep_size = 8
    num_local_experts = 34
    my_rank = 2
    go_py = my_rank * num_local_experts
    go_val = jnp.int32(go_py)
    me_val = jnp.int32(my_rank)
    dummy_gs = jnp.zeros((ep_size * num_local_experts,), dtype=jnp.int32)
    dummy_hdr = jnp.zeros((2 + ep_size + num_local_experts * ep_size,), dtype=jnp.int32)
    dummy_tr = jnp.int32(0)
    rot_env = "2" if rotate_rs else "0"

    rng = np.random.default_rng(2026 + t_global + int(rotate_rs))

    # Build 5 distributions covering all 3 tiers + localized macro-block overflow + 100% single-expert skew!
    cases: list[tuple[str, np.ndarray, set[int]]] = []

    # Case 1: Uniform -> Tier 0 (<= 32,768)
    cases.append(
        (
            "uniform_tier0",
            rng.integers(0, ep_size * num_local_experts, size=(t_global, 8), dtype=np.int32),
            {0},
        )
    )

    # Case 2: Medium chip skew (~42k at 24k, ~38k at 16k) -> Tier 1 (64k)
    med = rng.integers(0, ep_size * num_local_experts, size=(t_global, 8), dtype=np.int32)
    target_frac = 0.14 if t_global == 24576 else 0.20
    m_med = rng.random((t_global, 8)) < target_frac
    med = np.where(m_med, go_py + (med % num_local_experts), med).astype(np.int32)
    cases.append(("medium_skew_tier1", med, {1}))

    # Case 3: Localized macro-block 0 burst (> 40*128 = 5,120 local hits inside tokens 0..2047)
    # is dispersed by Stage-1 strided 8-row sampling out of Tier 2 into the minimum viable tier
    # (`Tier 0` when loc_total <= 32,768 at T=16K, `Tier 1` when loc_total ~34.7k > 32,768 at T=24K).
    burst = rng.integers(0, ep_size * num_local_experts, size=(t_global, 8), dtype=np.int32)
    burst[:1500, :] = go_py + (burst[:1500, :] % num_local_experts)
    cases.append(
        ("localized_burst_dispersed_below_tier2", burst, {0 if t_global == 16384 else 1})
    )

    # Case 4: Heavy > 64k global skew (~80k local hits) -> Tier 2
    heavy = rng.integers(0, ep_size * num_local_experts, size=(t_global, 8), dtype=np.int32)
    m_hvy = rng.random((t_global, 8)) < 0.45
    heavy = np.where(m_hvy, go_py + (heavy % num_local_experts), heavy).astype(np.int32)
    cases.append(("heavy_skew_tier2", heavy, {2}))

    # Case 5: 100% single-expert adversarial skew (all T_global * 8 entries -> Expert go_py!) -> Tier 2
    all_one = np.full((t_global, 8), go_py, dtype=np.int32)
    cases.append(("all_single_expert_100pct", all_one, {2}))

    seen_tiers: set[int] = set()
    for name, arr_np, expected_tiers in cases:
        g_ids = jnp.asarray(arr_np)
        rel = arr_np.reshape(-1) - go_py
        loc_total = int(np.sum((rel >= 0) & (rel < num_local_experts)))

        with (
            _tpu7x_cpu_context(interpret=True),
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                    "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                    "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": rot_env,
                    "SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT": "0",
                },
            ),
        ):
            lhs_ref, *_ = compute_block_scatter_routing(
                g_ids,
                go_val,
                me_val,
                ep_size=ep_size,
                num_local_experts=num_local_experts,
                topk=8,
                precomputed_counts=(dummy_gs, dummy_hdr, dummy_tr),
            )
            _, _, tier_val = pallas_compact_local_packed_keys(
                g_ids,
                go_val,
                me_val,
                num_local_experts=num_local_experts,
                ep_size=ep_size,
                rotate_rs=rotate_rs,
            )

        tier_int = int(tier_val)
        assert tier_int in expected_tiers, (
            f"{name} tg={t_global}: expected tier in {expected_tiers}, got {tier_int} "
            f"(loc_total={loc_total})"
        )
        seen_tiers.add(tier_int)
        lhs_ref_np = np.asarray(lhs_ref)

        for compact_mode in (1, 2):
            for counts_arg in ((dummy_gs, dummy_hdr, dummy_tr), None):
                if rotate_rs and counts_arg is None:
                    continue
                with (
                    _tpu7x_cpu_context(interpret=True),
                    mock.patch.dict(
                        os.environ,
                        {
                            "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                            "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                            "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": rot_env,
                            "SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT": str(compact_mode),
                        },
                    ),
                ):
                    lhs_got, *_ = compute_block_scatter_routing(
                        g_ids,
                        go_val,
                        me_val,
                        ep_size=ep_size,
                        num_local_experts=num_local_experts,
                        topk=8,
                        precomputed_counts=counts_arg,
                    )
                lhs_got_np = np.asarray(lhs_got)
                assert lhs_got_np.shape == lhs_ref_np.shape
                # Every single index (including padded tail [loc_total:]) must be strictly in-bounds [0, t_global)!
                assert int(lhs_got_np.min()) >= 0
                assert int(lhs_got_np.max()) < t_global
                assert np.array_equal(
                    lhs_ref_np[:loc_total], lhs_got_np[:loc_total]
                ), (
                    f"Active lhs_indices mismatch on {name} tg={t_global} "
                    f"mode={compact_mode} rot={rotate_rs} loc_total={loc_total}"
                )

    assert seen_tiers == {0, 1, 2}


def test_real_libtpu_tpu7x_aot_mosaic_compile():
    """Compile both new routing kernels with real libtpu.so on tpu7x=2x2x1 to catch any Mosaic codegen issue."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.experimental import topologies

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_compact_local_packed_keys,
        pallas_local_inv_perm_and_counts,
    )

    try:
        topo = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x=2x2x1")
    except Exception as exc:
        pytest.skip(f"tpu7x AOT topology unavailable: {exc}")

    shd = jax.sharding.SingleDeviceSharding(topo.devices[0])
    with _tpu7x_cpu_context(interpret=False):
        c_route = (
            jax.jit(
                lambda l, g, go, me: pallas_local_inv_perm_and_counts(
                    l,
                    g,
                    go,
                    me,
                    ep_size=8,
                    num_local_experts=34,
                    total_id_buckets=273,
                    has_pad_sentinel=True,
                    native_pack=True,
                )
            )
            .trace(
                jax.ShapeDtypeStruct((3072, 8), jnp.int32, sharding=shd),
                jax.ShapeDtypeStruct((24576, 8), jnp.int32, sharding=shd),
                jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
                jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
            )
            .lower()
            .compile()
        )
        assert c_route is not None

        for rot in (False, True):
            c_comp = (
                jax.jit(
                    lambda g, go, me: pallas_compact_local_packed_keys(
                        g,
                        go,
                        me,
                        num_local_experts=34,
                        ep_size=8,
                        rotate_rs=rot,
                    )
                )
                .trace(
                    jax.ShapeDtypeStruct((24576, 8), jnp.int32, sharding=shd),
                    jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
                    jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
                )
                .lower()
                .compile()
            )
            assert c_comp is not None

        c_comp_ep16 = (
            jax.jit(
                lambda g, go, me: pallas_compact_local_packed_keys(
                    g,
                    go,
                    me,
                    num_local_experts=18,
                    ep_size=16,
                    rotate_rs=True,
                )
            )
            .trace(
                jax.ShapeDtypeStruct((24576, 8), jnp.int32, sharding=shd),
                jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
                jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
            )
            .lower()
            .compile()
        )
        assert c_comp_ep16 is not None


def test_strided_compact_qblock_r4_tier0_and_resolve_qblock_non_po2():
    """Verify q_block R=4 spatial concentration compacts into Tier 0 (0-ULP) and non-Po2 T>32K candidates."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        compute_block_scatter_routing,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_compact_local_packed_keys,
        resolve_qblock_params,
    )

    assert resolve_qblock_params(36864 * 8, 8, 256) == (96, 128)
    assert resolve_qblock_params(36864 * 8, 16, 256) == (96, 128)

    t_global, ep_size, num_local_experts = 24576, 8, 36
    rng = np.random.default_rng(777)
    g_np = rng.integers(0, ep_size * num_local_experts, size=(t_global, 8), dtype=np.int32)
    # Emulate R=4 q_block contiguous slab assignment: concentrate replica slots into tokens 0..6143
    # (exceeding old Stage-1 r_keep1=40 and Stage-2 quarter=8,192 limits while keeping rank total < 32K).
    g_np = np.where((g_np < num_local_experts), num_local_experts + (g_np % num_local_experts), g_np)
    g_np[:6144, :4] = rng.integers(0, num_local_experts, size=(6144, 4), dtype=np.int32)
    loc_total = int(np.sum(g_np < num_local_experts))
    assert loc_total <= 32768

    g_ids = jnp.asarray(g_np)
    go_val = jnp.int32(0)
    me_val = jnp.int32(0)
    dummy_gs = jnp.zeros((ep_size * num_local_experts,), dtype=jnp.int32)
    dummy_hdr = jnp.zeros((2 + ep_size + num_local_experts * ep_size,), dtype=jnp.int32)
    dummy_tr = jnp.int32(0)

    with (
        _tpu7x_cpu_context(interpret=True),
        mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT": "0",
            },
        ),
    ):
        lhs_ref, *_ = compute_block_scatter_routing(
            g_ids,
            go_val,
            me_val,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            topk=8,
            precomputed_counts=(dummy_gs, dummy_hdr, dummy_tr),
        )

    with (
        _tpu7x_cpu_context(interpret=True),
        mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                "SGLANG_TPU_MUSE_KERNEL_GMM_ROTATE_RS": "2",
                "SGLANG_TPU_MUSE_KERNEL_COMPACT_LOCAL_SORT": "1",
            },
        ),
    ):
        _, _, tier_val = pallas_compact_local_packed_keys(
            g_ids,
            go_val,
            me_val,
            num_local_experts=num_local_experts,
            ep_size=ep_size,
            rotate_rs=True,
        )
        lhs_got, *_ = compute_block_scatter_routing(
            g_ids,
            go_val,
            me_val,
            ep_size=ep_size,
            num_local_experts=num_local_experts,
            topk=8,
            precomputed_counts=(dummy_gs, dummy_hdr, dummy_tr),
        )

    assert int(tier_val) == 0
    assert np.array_equal(np.asarray(lhs_ref)[:loc_total], np.asarray(lhs_got)[:loc_total])


def test_macro_vec_qblock_coarse48_parity_and_tpu7x_aot():
    """Verify macro-vectorized q_block (n_blocks=48 at 24K, 64 at 8K/16K/32K) is race-free 0-ULP and compiles."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax import export
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.muse_prefill_kernels.eplb import remap_zeroglue_virtual_ids
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_qblock_remap_virtual_ids,
        resolve_qblock_params,
    )

    assert resolve_qblock_params(8192 * 8, 8, 256) == (64, 128)
    assert resolve_qblock_params(12288 * 8, 8, 256) == (48, 128)
    assert resolve_qblock_params(16384 * 8, 8, 256) == (64, 128)
    assert resolve_qblock_params(24576 * 8, 8, 256) == (48, 128)
    assert resolve_qblock_params(24576 * 8, 16, 256) == (48, 128)
    assert resolve_qblock_params(32768 * 8, 8, 256) == (64, 128)
    assert resolve_qblock_params(36864 * 8, 8, 256) == (96, 128)

    rng = np.random.default_rng(20261126)
    eager_interp = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    cases = (
        (8, 32, 4096, 128, 4, 2, True),
        (8, 32, 8192, 128, 4, 5, True),
        (8, 32, 16384, 256, 2, 1, False),
        (8, 32, 24576, 256, 4, 7, True),
        (8, 32, 24576, 256, 1, 0, False),
        (16, 16, 8192, 128, 2, 11, True),
        (16, 16, 24576, 256, 2, 0, False),
        (16, 16, 24576, 256, 4, 15, True),
        (16, 16, 24576, 256, 12, 15, True),
        (16, 16, 24576, 256, 16, 0, False),
    )
    with _tpu7x_cpu_context(interpret=False):
        for ep_size, base_loc, t_tot, q_m, n_rep, rk, add_pad in cases:
            e_loc = base_loc + n_rep
            probs = 1.0 / (np.arange(1, ep_size * base_loc + 1, dtype=np.float64) ** 1.3)
            probs /= probs.sum()
            g_np = rng.choice(ep_size * base_loc, size=(t_tot, 8), p=probs).astype(np.int32)
            if add_pad:
                g_np[-128:, :] = ep_size * base_loc
            g_j = jnp.asarray(g_np)
            rk_j = jnp.int32(rk)
            with mock.patch.dict(
                os.environ, {"SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "0"}
            ):
                ref_g, ref_l = remap_zeroglue_virtual_ids(
                    g_j,
                    rk_j,
                    ep_size=ep_size,
                    num_local_experts=e_loc,
                    num_base_local=base_loc,
                    mode="q_block",
                    quantum=q_m,
                )
            got_g, got_l = pallas_qblock_remap_virtual_ids(
                g_j,
                rk_j,
                ep_size=ep_size,
                num_local_experts=e_loc,
                num_base_local=base_loc,
                quantum=q_m,
                interpret=eager_interp,
            )
            assert np.array_equal(np.asarray(got_g), np.asarray(ref_g)), (
                f"global mismatch ep={ep_size} tg={t_tot} rep={n_rep} rk={rk}"
            )
            assert np.array_equal(np.asarray(got_l), np.asarray(ref_l)), (
                f"local mismatch ep={ep_size} tg={t_tot} rep={n_rep} rk={rk}"
            )

        exp = export.export(
            jax.jit(
                lambda g, me: pallas_qblock_remap_virtual_ids(
                    g,
                    me,
                    ep_size=8,
                    num_local_experts=36,
                    num_base_local=32,
                    quantum=256,
                )
            ),
            platforms=("tpu",),
        )(
            jax.ShapeDtypeStruct((24576, 8), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
        )
        assert "pallas_qblock_eplb_remap_tg24576_ep8_el36_nb48_q128" in exp.mlir_module()

        for topo_name, ep_size, base_loc, e_loc in (
            ("tpu7x=2x2x1", 8, 32, 36),
            ("tpu7x=2x2x2", 16, 16, 18),
            ("tpu7x=2x2x2", 16, 16, 28),
            ("tpu7x=2x2x2", 16, 16, 32),
        ):
            topo = topologies.get_topology_desc(platform="tpu", topology_name=topo_name)
            shd = jax.sharding.SingleDeviceSharding(topo.devices[0])
            c_qb = (
                jax.jit(
                    lambda g, me, _ep=ep_size, _bl=base_loc, _el=e_loc: pallas_qblock_remap_virtual_ids(
                        g,
                        me,
                        ep_size=_ep,
                        num_local_experts=_el,
                        num_base_local=_bl,
                        quantum=256,
                    )
                )
                .trace(
                    jax.ShapeDtypeStruct((24576, 8), jnp.int32, sharding=shd),
                    jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
                )
                .lower()
                .compile()
            )
            assert c_qb is not None

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _allow_pallas_sortfree_routing,
    )

    assert not _allow_pallas_sortfree_routing(
        1,
        sortfree_prefill=True,
        topk=8,
        ep_size=8,
        t_local=64,
        n_loc_eff=32,
        has_pad_sentinel=True,
        eff_hoist_routing=True,
        eff_cm_pipe=False,
    )
    assert _allow_pallas_sortfree_routing(
        1,
        sortfree_prefill=True,
        topk=8,
        ep_size=8,
        t_local=128,
        n_loc_eff=32,
        has_pad_sentinel=True,
        eff_hoist_routing=True,
        eff_cm_pipe=False,
    )
    assert _allow_pallas_sortfree_routing(
        1,
        sortfree_prefill=True,
        topk=8,
        ep_size=8,
        t_local=256,
        n_loc_eff=32,
        has_pad_sentinel=True,
        eff_hoist_routing=True,
        eff_cm_pipe=False,
    )


def test_fori_part_b_ep16_and_ep8_parity_and_tpu7x_aot():
    """Verify lax.fori_loop Part B in pallas_local_inv_perm_and_counts (EP=16/8)."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _allow_pallas_sortfree_routing,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_local_inv_perm_and_counts,
    )

    eager_interp = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(20261127)
    cases = (
        # (ep, tl, n_loc, rk, has_pad, rnp, zls)
        (16, 128, 18, 0, True, False, True),
        (16, 128, 18, 15, True, True, True),
        (16, 256, 16, 3, False, False, True),
        (16, 256, 16, 7, False, True, False),
        (16, 256, 18, 0, True, False, True),
        (16, 256, 18, 11, True, True, True),
        (16, 256, 32, 15, True, True, False),
        (16, 512, 22, 9, True, True, True),
        (8, 256, 36, 4, True, True, False),
    )
    with _tpu7x_cpu_context(interpret=False):
        for ep_size, t_local, n_loc, rk, has_pad, rnp, zls in cases:
            t_glob = ep_size * t_local
            n_exp = ep_size * n_loc
            nb = n_exp + (1 if has_pad else 0)
            g_np = rng.integers(0, n_exp, size=(t_glob, 8), dtype=np.int32)
            if has_pad:
                vt = max(1, (rk + 1) * t_local - 37)
                g_np[vt:] = n_exp
            l_np = g_np[rk * t_local : (rk + 1) * t_local]
            ip, gs, gst, tr = pallas_local_inv_perm_and_counts(
                jnp.asarray(l_np),
                jnp.asarray(g_np),
                jnp.int32(rk * n_loc),
                jnp.int32(rk),
                ep_size=ep_size,
                num_local_experts=n_loc,
                total_id_buckets=nb,
                has_pad_sentinel=has_pad,
                zero_local_start=zls,
                native_pack=rnp,
                interpret=eager_interp,
            )
            exp_ip = (
                np.argsort(np.argsort(l_np.reshape(-1), kind="stable"))
                .reshape(t_local, 8)
                .astype(np.int32)
            )
            assert np.array_equal(np.asarray(ip), exp_ip)
            g_chips = g_np.reshape(ep_size, t_local * 8)
            exp_bef = np.sum(g_chips < (rk * n_loc), axis=1).astype(np.int32)
            exp_ce = np.stack(
                [np.sum(g_chips == (rk * n_loc + e), axis=1) for e in range(n_loc)],
                axis=0,
            ).astype(np.int32)
            exp_ls = np.int32(0) if zls else np.int32(exp_bef.sum())
            act_rk = int(np.sum(g_chips[rk] < n_exp)) if has_pad else (t_local * 8)
            exp_tr = np.int32(act_rk - int(exp_ce[:, rk].sum()))
            exp_hdr = np.concatenate(
                [
                    np.asarray([exp_ls, exp_tr], dtype=np.int32),
                    exp_bef,
                    exp_ce.reshape(-1),
                ]
            )
            exp_gs_loc = np.sum(exp_ce, axis=1).astype(np.int32)
            assert np.array_equal(
                np.asarray(gs)[rk * n_loc : (rk + 1) * n_loc], exp_gs_loc
            )
            assert np.array_equal(np.asarray(gst), exp_hdr)
            assert int(tr) == int(exp_tr)

        orig_cp = pltpu.CompilerParams
        with mock.patch.object(
            pltpu,
            "CompilerParams",
            lambda *a, **kw: orig_cp(*a, **{**kw, "disable_bounds_checks": False}),
        ):
            for topo_name, ep_size, tl, n_loc, rnp in (
                ("tpu7x=2x2x2", 16, 128, 18, False),
                ("tpu7x=2x2x2", 16, 128, 18, True),
                ("tpu7x=2x2x1", 8, 128, 32, False),
                ("tpu7x=2x2x2", 16, 256, 16, False),
                ("tpu7x=2x2x2", 16, 256, 18, True),
                ("tpu7x=2x2x2", 16, 512, 22, True),
                ("tpu7x=2x2x2", 16, 1536, 32, True),
                ("tpu7x=2x2x1", 8, 256, 36, True),
            ):
                topo = topologies.get_topology_desc(platform="tpu", topology_name=topo_name)
                shd = jax.sharding.SingleDeviceSharding(topo.devices[0])
                fn_aot = (
                    lambda l_, g_, go_, m_, _ep=ep_size, _nl=n_loc, _r=rnp: (
                        pallas_local_inv_perm_and_counts(
                            l_,
                            g_,
                            go_,
                            m_,
                            ep_size=_ep,
                            num_local_experts=_nl,
                            total_id_buckets=_ep * _nl + 1,
                            has_pad_sentinel=True,
                            zero_local_start=True,
                            native_pack=_r,
                        )
                    )
                )
                comp = (
                    jax.jit(fn_aot)
                    .trace(
                        jax.ShapeDtypeStruct((tl, 8), jnp.int32, sharding=shd),
                        jax.ShapeDtypeStruct((ep_size * tl, 8), jnp.int32, sharding=shd),
                        jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
                        jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
                    )
                    .lower()
                    .compile()
                )
                assert comp is not None

    for pad in (False, True):
        assert _allow_pallas_sortfree_routing(
            1, sortfree_prefill=True, topk=8, ep_size=16, t_local=256,
            n_loc_eff=22, has_pad_sentinel=pad, eff_hoist_routing=True, eff_cm_pipe=False,
        )
    assert _allow_pallas_sortfree_routing(
        1, sortfree_prefill=True, topk=8, ep_size=16, t_local=128,
        n_loc_eff=18, has_pad_sentinel=True, eff_hoist_routing=True, eff_cm_pipe=True,
    )
    assert not _allow_pallas_sortfree_routing(
        1, sortfree_prefill=True, topk=8, ep_size=16, t_local=64,
        n_loc_eff=18, has_pad_sentinel=True, eff_hoist_routing=True, eff_cm_pipe=True,
    )


def test_w19_inv_perm_zax1_and_qblock_macro32_64_parity_and_aot():
    """Verify Wave-19 0-axis1-reduce inv_perm (incl. duplicate ties) and macro32/64 q_block."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax import export
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.muse_prefill_kernels.eplb import remap_zeroglue_virtual_ids
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_local_inv_perm_and_counts,
        pallas_qblock_remap_virtual_ids,
    )

    eager_interp = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(20261137)
    with _tpu7x_cpu_context(interpret=False):
        # 1. pallas_qblock_remap_virtual_ids macro32 (tg=4096) & macro64 (tg=2048) 0-ULP parity.
        for ep_sz, base_loc, t_tot, n_rep, rk, add_pad in (
            (16, 16, 2048, 6, 3, True),
            (16, 16, 4096, 6, 11, True),
            (8, 32, 4096, 4, 5, False),
        ):
            e_loc = base_loc + n_rep
            g_np = rng.integers(0, ep_sz * base_loc, size=(t_tot, 8), dtype=np.int32)
            if add_pad:
                g_np[-128:, :] = ep_sz * base_loc
            g_j, rk_j = jnp.asarray(g_np), jnp.int32(rk)
            with mock.patch.dict(
                os.environ, {"SGLANG_TPU_MUSE_KERNEL_PALLAS_SORTFREE_ROUTING": "0"}
            ):
                ref_g, ref_l = remap_zeroglue_virtual_ids(
                    g_j, rk_j, ep_size=ep_sz, num_local_experts=e_loc,
                    num_base_local=base_loc, mode="q_block", quantum=128,
                )
            got_g, got_l = pallas_qblock_remap_virtual_ids(
                g_j, rk_j, ep_size=ep_sz, num_local_experts=e_loc,
                num_base_local=base_loc, quantum=128, interpret=eager_interp,
            )
            assert np.array_equal(np.asarray(got_g), np.asarray(ref_g))
            assert np.array_equal(np.asarray(got_l), np.asarray(ref_l))

        for tg, ep_sz, bl, el, topo_name in (
            (2048, 16, 16, 22, "tpu7x=2x2x2"),
            (4096, 16, 16, 22, "tpu7x=2x2x2"),
            (4096, 8, 32, 36, "tpu7x=2x2x1"),
        ):
            exp_qb = export.export(
                jax.jit(
                    lambda g, me, _ep=ep_sz, _bl=bl, _el=el: pallas_qblock_remap_virtual_ids(
                        g, me, ep_size=_ep, num_local_experts=_el, num_base_local=_bl, quantum=128
                    )
                ),
                platforms=("tpu",),
            )(jax.ShapeDtypeStruct((tg, 8), jnp.int32), jax.ShapeDtypeStruct((), jnp.int32))
            assert f"pallas_qblock_eplb_remap_tg{tg}_ep{ep_sz}_el{el}_nb64_q128" in exp_qb.mlir_module()
            topo = topologies.get_topology_desc(platform="tpu", topology_name=topo_name)
            shd = jax.sharding.SingleDeviceSharding(topo.devices[0])
            comp_qb = (
                jax.jit(
                    lambda g, me, _ep=ep_sz, _bl=bl, _el=el: pallas_qblock_remap_virtual_ids(
                        g, me, ep_size=_ep, num_local_experts=_el, num_base_local=_bl, quantum=128
                    )
                )
                .trace(
                    jax.ShapeDtypeStruct((tg, 8), jnp.int32, sharding=shd),
                    jax.ShapeDtypeStruct((), jnp.int32, sharding=shd),
                )
                .lower()
                .compile()
            )
            assert comp_qb is not None

        # 2. pallas_local_inv_perm_and_counts zax1 duplicate-slot adversarial parity (has_pad=False).
        g_dup = np.full((16 * 128, 8), 5 * 22 + 3, dtype=np.int32)
        g_dup[:64, 0] = 0
        g_dup[5 * 128 + 10, :] = 5 * 22 + 3
        l_dup = g_dup[5 * 128 : 6 * 128]
        ip_d, gs_d, gst_d, tr_d = pallas_local_inv_perm_and_counts(
            jnp.asarray(l_dup), jnp.asarray(g_dup), jnp.int32(5 * 22), jnp.int32(5),
            ep_size=16, num_local_experts=22, total_id_buckets=16 * 22,
            has_pad_sentinel=False, zero_local_start=True, native_pack=True,
            interpret=eager_interp,
        )
        exp_dup = np.argsort(np.argsort(l_dup.reshape(-1), kind="stable")).reshape(128, 8).astype(np.int32)
        assert np.array_equal(np.asarray(ip_d), exp_dup)
        assert int(tr_d) == int(gst_d[1])
        assert int(gs_d[5 * 22 + 3]) == int(np.sum(g_dup == (5 * 22 + 3)))
