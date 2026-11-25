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
