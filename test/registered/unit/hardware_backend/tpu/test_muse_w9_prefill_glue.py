# SPDX-License-Identifier: Apache-2.0
"""Wave-9 prefill collectives, layout, and glue tests (u2x2x2 + u2x2x1)."""

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

_TOUCHED_FLAGS = (
    "SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG",
    "SGLANG_TPU_MUSE_KERNEL_XOR1_AG_MOE",
    "SGLANG_TPU_MUSE_KERNEL_XOR1_AG_CHUNK",
    "SGLANG_TPU_MUSE_KERNEL_XOR1_AG_UNDER_D",
    "SGLANG_TPU_MUSE_KERNEL_XOR1_AG_B2",
    "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT",
    "SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1",
)


@contextlib.contextmanager
def _interpret_on_cpu():
    import jax
    from jax.experimental import pallas as pl

    real = pl.pallas_call

    def _interpret(*args, **kwargs):
        kwargs["interpret"] = True
        return real(*args, **kwargs)

    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
        mock.patch.object(pl, "pallas_call", _interpret),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        yield


def test_w9_prefill_glue_flags_default_off_and_ast_get_hygiene():
    """All prefill glue flags default off and every sglang_tpu EnvVar is read via .get()."""
    import re

    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_MOE.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_CHUNK.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_UNDER_D.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_XOR1_AG_B2.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1.get() == 0

    repo = pathlib.Path(__file__).resolve().parents[5]
    rel_paths = (
        "sglang_tpu/kernels/xor1_bridge_all_gather.py",
        "sglang_tpu/muse_prefill_kernels/attn_kernels/kernel_a1.py",
        "sglang_tpu/muse_prefill_kernels/integration.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/pipeline.py",
    )
    checked = set(_TOUCHED_FLAGS)
    for rel in rel_paths:
        tree = ast.parse((repo / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in checked:
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

    declared: set[str] = set()
    for env_rel in ("sglang_tpu/envs.py", "python/sglang/srt/environ.py"):
        env_p = repo / env_rel
        if env_p.exists():
            declared.update(
                re.findall(r"^\s*(SGLANG_TPU_[A-Z0-9_]+)\s*=\s*Env", env_p.read_text(), re.M)
            )
    assert declared

    def _is_env_ref(node: ast.AST, local_aliases: set[str] | None = None) -> str | None:
        if isinstance(node, ast.Attribute) and node.attr in declared:
            if isinstance(node.value, ast.Name) and node.value.id in (
                "envs",
                "tpu_envs",
                "_tpu_envs",
                "_envs_mod",
            ):
                return node.attr
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in declared
        ):
            return str(node.args[1].value)
        if local_aliases and isinstance(node, ast.Name) and node.id in local_aliases:
            return f"alias:{node.id}"
        return None

    violations: list[str] = []
    for py_path in sorted((repo / "sglang_tpu").rglob("*.py")):
        tree = ast.parse(py_path.read_text(errors="replace"), filename=str(py_path))
        # Track per-module assigned EnvVar local aliases (e.g. `_e = getattr(envs, "SGLANG_TPU_...", None)`).
        local_aliases: set[str] = set()
        for stmt in ast.walk(tree):
            if isinstance(stmt, ast.Assign) and _is_env_ref(stmt.value) is not None:
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        local_aliases.add(tgt.id)
        for node in ast.walk(tree):
            candidates: list[ast.AST] = []
            if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert)):
                candidates.append(node.test)
            elif isinstance(node, ast.BoolOp):
                candidates.extend(node.values)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                candidates.append(node.operand)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("bool", "int", "float", "str")
                and node.args
            ):
                candidates.append(node.args[0])
            elif isinstance(node, ast.Compare):
                is_none_cmp = all(
                    isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops
                ) and any(
                    isinstance(c, ast.Constant) and c.value is None
                    for c in node.comparators
                )
                if not is_none_cmp:
                    candidates.append(node.left)
                    candidates.extend(node.comparators)
            for cand in candidates:
                env_name = _is_env_ref(cand, local_aliases=local_aliases)
                if env_name is not None:
                    rel = py_path.relative_to(repo)
                    violations.append(
                        f"{rel}:{getattr(cand, 'lineno', 0)} ({env_name})"
                    )
    assert not violations, "EnvVar read without .get() in boolean/cast/compare context: " + ", ".join(violations)


def test_xor1_xyz_torus_ag_schedule_simulation_and_lowering():
    """Verify 16-rank 1-hop XYZ torus + XOR=1 D2D schedule correctness and TPU7x HLO."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.kernels.xor1_bridge_all_gather import (
        _xyz_row_splits,
        xor1_ag_min_row_align,
        xor1_ag_schedule_spec,
        xor1_bridge_all_gather,
    )

    for rows, align in ((512, 32), (256, 32), (384, 8), (128, 32), (96, 32)):
        r0, r1, r2 = _xyz_row_splits(rows, align)
        assert r0 + r1 + r2 == rows and min(r0, r1, r2) >= align
        slices = ((0, r0), (r0, r0 + r1), (r0 + r1, rows))
        rng = np.random.default_rng(rows)
        x = rng.integers(-120, 120, size=(16, rows, 4), dtype=np.int8)
        out = [
            [np.full((rows, 4), -999, dtype=np.int32) for _ in range(16)]
            for _ in range(16)
        ]
        for me in range(16):
            out[me][me] = x[me].astype(np.int32)
            out[me ^ 1][me] = x[me].astype(np.int32)

        def _exec_step(transfers):
            in_flight = []
            for me, owner, p_idx, dst in transfers:
                a, b = slices[p_idx]
                src = x[me] if owner is None else out[me][owner]
                assert not np.any(src[a:b] == -999)
                slot = me if owner is None else owner
                in_flight.append((dst, slot, a, b, src[a:b].copy()))
            for dst, slot, a, b, data in in_flight:
                assert np.all(out[dst][slot][a:b] == -999)
                out[dst][slot][a:b] = data

        _exec_step(
            [(me, None, 0, me ^ 2) for me in range(16)]
            + [(me, None, 1, me ^ 4) for me in range(16)]
            + [(me, None, 2, me ^ 8) for me in range(16)]
        )
        s2 = []
        for me in range(16):
            for owner in (None, me ^ 2):
                s2.append((me, owner, 0, me ^ 4))
            for owner in (None, me ^ 4):
                s2.append((me, owner, 1, me ^ 8))
            for owner in (None, me ^ 8):
                s2.append((me, owner, 2, me ^ 2))
            for idx, peer in enumerate((me ^ 2, me ^ 4, me ^ 8)):
                s2.append((me, peer, idx, me ^ 1))
        _exec_step(s2)
        s3 = []
        for me in range(16):
            for owner in (None, me ^ 2, me ^ 4, me ^ 6):
                s3.append((me, owner, 0, me ^ 8))
            for owner in (None, me ^ 4, me ^ 8, me ^ 12):
                s3.append((me, owner, 1, me ^ 2))
            for owner in (None, me ^ 8, me ^ 2, me ^ 10):
                s3.append((me, owner, 2, me ^ 4))
            for peer, p_idx in (
                (me ^ 4, 0),
                (me ^ 6, 0),
                (me ^ 8, 1),
                (me ^ 12, 1),
                (me ^ 2, 2),
                (me ^ 10, 2),
            ):
                s3.append((me, peer, p_idx, me ^ 1))
        _exec_step(s3)
        s4 = []
        for me in range(16):
            for peer in (me ^ 8, me ^ 10, me ^ 12, me ^ 14):
                s4.append((me, peer, 0, me ^ 1))
            for peer in (me ^ 2, me ^ 6, me ^ 10, me ^ 14):
                s4.append((me, peer, 1, me ^ 1))
            for peer in (me ^ 4, me ^ 12, me ^ 6, me ^ 14):
                s4.append((me, peer, 2, me ^ 1))
        _exec_step(s4)
        for r in range(16):
            np.testing.assert_array_equal(np.stack(out[r], axis=0), x.astype(np.int32))

    assert xor1_ag_min_row_align(4) == 8
    assert xor1_ag_min_row_align(2) == 16
    assert xor1_ag_min_row_align(1) == 32
    assert xor1_ag_schedule_spec(16, 512, 1, xy_torus=True)[0] == "xyz"
    assert xor1_ag_schedule_spec(16, 512, 2, xy_torus=True)[:2] == ("xyz", (176, 176, 160))
    assert xor1_ag_schedule_spec(16, 128, 2, xy_torus=True)[:2] == ("xyz", (48, 48, 32))
    assert xor1_ag_schedule_spec(8, 512, 1, xy_torus=True)[0] == "xy"
    assert xor1_ag_schedule_spec(8, 24, 2, xy_torus=True)[0] == "flat"
    assert xor1_ag_schedule_spec(8, 32, 2, xy_torus=True)[:2] == ("xy", (16, 16))
    assert xor1_ag_schedule_spec(16, 64, 1, xy_torus=True)[0] == "flat"
    assert xor1_ag_schedule_spec(16, 100, 1, xy_torus=True)[0] == "flat"

    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip("Requires >=8 JAX devices")

    mesh8 = Mesh(np.array(devs[:8]), ("model",))
    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
        mock.patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "3"}),
    ):
        sm8 = jax.jit(
            jax.shard_map(
                lambda x: xor1_bridge_all_gather(x[0], axis_name="model", tiled=False)[None],
                mesh=mesh8,
                in_specs=(P("model", None, None),),
                out_specs=P("model", None, None, None),
                check_vma=False,
            )
        )
        for rows, dt in ((512, jnp.float8_e4m3fn), (32, jnp.bfloat16), (16, jnp.float32)):
            hlo8 = (
                sm8.trace(jax.ShapeDtypeStruct((8, rows, 128), dt))
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )
            assert "xor1_xytorus_ag_tp8" in hlo8 and "partition_id" not in hlo8
        hlo8_lax = (
            sm8.trace(jax.ShapeDtypeStruct((8, 24, 128), jnp.bfloat16))
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert "all_gather" in hlo8_lax and "xor1_" not in hlo8_lax
        hlo8_flat = (
            sm8.trace(jax.ShapeDtypeStruct((8, 48, 128), jnp.bfloat16))
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert "xor1_bridge_ag_tp8" in hlo8_flat and "partition_id" not in hlo8_flat

        if len(devs) >= 16:
            mesh16 = Mesh(np.array(devs[:16]), ("model",))
            sm16 = jax.jit(
                jax.shard_map(
                    lambda x: xor1_bridge_all_gather(x[0], axis_name="model", tiled=False)[None],
                    mesh=mesh16,
                    in_specs=(P("model", None, None),),
                    out_specs=P("model", None, None, None),
                    check_vma=False,
                )
            )
            from jax.experimental import topologies

            try:
                topo16 = topologies.get_topology_desc(
                    platform="tpu", topology_name="tpu7x=2x2x2"
                )
                aot_mesh16 = Mesh(np.array(topo16.devices).reshape(16), ("model",))
                aot_sm16 = jax.jit(
                    jax.shard_map(
                        lambda x: xor1_bridge_all_gather(
                            x[0], axis_name="model", tiled=False
                        )[None],
                        mesh=aot_mesh16,
                        in_specs=(P("model", None, None),),
                        out_specs=P("model", None, None, None),
                        check_vma=False,
                    )
                )
            except Exception:
                aot_mesh16 = None
                aot_sm16 = None

            for dt, item_bytes in (
                (jnp.float32, 4),
                (jnp.bfloat16, 2),
                (jnp.float8_e4m3fn, 1),
            ):
                align = xor1_ag_min_row_align(item_bytes)
                for rows in (128, 512, 1280, 4096):
                    kind, splits, _ = xor1_ag_schedule_spec(
                        16, rows, item_bytes, xy_torus=True
                    )
                    assert kind == "xyz"
                    assert sum(splits) == rows and all(
                        s >= align and s % align == 0 for s in splits
                    )
                    hlo16 = (
                        sm16.trace(jax.ShapeDtypeStruct((16, rows, 128), dt))
                        .lower(lowering_platforms=("tpu",))
                        .as_text("stablehlo")
                    )
                    assert "xor1_xyztorus_ag_tp16" in hlo16 and "partition_id" not in hlo16
                    if aot_sm16 is not None and aot_mesh16 is not None:
                        shd16 = jax.sharding.NamedSharding(
                            aot_mesh16, P("model", None, None)
                        )
                        compiled = (
                            aot_sm16.trace(
                                jax.ShapeDtypeStruct((16, rows, 128), dt, sharding=shd16)
                            )
                            .lower()
                            .compile()
                        )
                        assert compiled is not None

            # 16-device Mosaic interpret=True + detect_races=True parity for unequal (64, 32, 32) rows=128 split.
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            from jax.sharding import NamedSharding

            orig_pc = pl.pallas_call

            def _mosaic_interp(*args, **kwargs):
                kwargs["interpret"] = pltpu.InterpretParams(
                    detect_races=True, uninitialized_memory="nan"
                )
                return orig_pc(*args, **kwargs)

            rng128 = np.random.default_rng(128)
            x128_np = rng128.integers(-50, 50, size=(16, 128, 256), dtype=np.int8)
            x128_j = jax.device_put(
                jnp.asarray(x128_np).astype(jnp.float8_e4m3fn),
                NamedSharding(mesh16, P("model", None, None)),
            )
            with mock.patch.object(pl, "pallas_call", _mosaic_interp):
                out128 = jax.jit(
                    jax.shard_map(
                        lambda s: xor1_bridge_all_gather(
                            s[0], axis_name="model", tiled=False
                        )[None],
                        mesh=mesh16,
                        in_specs=(P("model", None, None),),
                        out_specs=P("model", None, None, None),
                        check_vma=False,
                    )
                )(x128_j)
            for r in range(16):
                np.testing.assert_array_equal(np.asarray(out128)[r], np.asarray(x128_j))


def test_tp16_a1_rpa_layout_parity_and_lowering():
    """Verify TP=16 (h_q=8, h_kv=1) A1_RPA_LAYOUT=1 matches prepare_inputs 0-ULP + lowers cleanly."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import prepare_inputs
    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1

    devs = jax.devices()
    if len(devs) < 16:
        pytest.skip("Requires 16 JAX devices")

    mesh = Mesh(np.array(devs[:16]), ("model",))
    tp_d, c_k, h_q_local, h_kv_local, hidden = 16, 128, 8, 1, 8192
    q_cols = h_q_local * 64
    qg_cols = 2 * q_cols + 128
    c_tok = tp_d * c_k

    def _make_sm(rpa_layout: bool):
        def _body(ag_k, wqkvg, wsc, cos_k, sin_k):
            q_k, g_k, kv_k = kernel_a1.pallas_a1_chunk_1way(
                ag_k,
                wqkvg,
                wsc,
                cos_k.reshape(tp_d, c_k, 128),
                sin_k.reshape(tp_d, c_k, 128),
                None,
                t_shard=c_k,
                row_off=0,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=rpa_layout,
                mode="all",
                tp_axis="model",
                tm=128,
            )
            if rpa_layout:
                return (
                    q_k.reshape(1, c_tok, 4, 2, 128),
                    g_k.reshape(c_tok, q_cols),
                    kv_k.reshape(c_tok, 1, 4, 128),
                )
            return (
                q_k.reshape(c_tok, q_cols),
                g_k.reshape(c_tok, q_cols),
                kv_k.reshape(c_tok, 128),
            )

        return jax.jit(
            jax.shard_map(
                _body,
                mesh=mesh,
                in_specs=(P("model"),) * 5,
                out_specs=(P("model"),) * 3,
                check_vma=False,
            )
        )

    rng = np.random.default_rng(77)
    ag_j = jnp.asarray((rng.standard_normal((tp_d * tp_d, c_k, hidden)) * 0.2).astype(np.float32)).astype(jnp.float8_e4m3fn)
    w_j = jnp.asarray((rng.standard_normal((tp_d * hidden, qg_cols)) * 0.05).astype(np.float32)).astype(jnp.float8_e4m3fn)
    wsc_j = jnp.asarray(rng.uniform(0.01, 0.05, size=(tp_d, qg_cols)).astype(np.float32))
    cos_j = jnp.asarray(np.cos(rng.uniform(-1, 1, size=(tp_d * tp_d * c_k, 128))).astype(np.float32))
    sin_j = jnp.asarray(np.sin(rng.uniform(-1, 1, size=(tp_d * tp_d * c_k, 128))).astype(np.float32))

    for sub_n, fast_norm in (("0", "0"), ("256", "1")):
        with (
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_A1_SUB_N": sub_n,
                    "SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM": fast_norm,
                },
            ),
            _interpret_on_cpu(),
            mock.patch(
                "jax.experimental.pallas.tpu.roll",
                side_effect=lambda a, shift, axis: jnp.roll(a, shift=shift, axis=axis),
            ),
        ):
            q_u, g_u, kv_u = _make_sm(False)(ag_j, w_j, wsc_j, cos_j, sin_j)
            q_r, g_r, kv_r = _make_sm(True)(ag_j, w_j, wsc_j, cos_j, sin_j)
        q_ref, kv_ref, _ = prepare_inputs(
            q_u[:c_tok].reshape(c_tok, h_q_local, 64),
            kv_u[:c_tok, :64].reshape(c_tok, 1, 64),
            kv_u[:c_tok, 64:].reshape(c_tok, 1, 64),
        )
        np.testing.assert_array_equal(
            np.asarray(g_r[:c_tok].view(np.uint16)),
            np.asarray(g_u[:c_tok].view(np.uint16)),
        )
        q_r_f32 = np.asarray(q_r[0].astype(jnp.float32))
        np.testing.assert_array_equal(q_r_f32, np.asarray(q_ref[0].astype(jnp.float32)))
        assert np.all(q_r_f32[..., 64:] == 0.0)
        np.testing.assert_array_equal(
            np.asarray(kv_r[:c_tok].view(np.uint8)),
            np.asarray(kv_ref.view(np.uint8)),
        )

    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
        mock.patch.dict(
            os.environ,
            {"SGLANG_TPU_MUSE_KERNEL_A1_SUB_N": "256", "SGLANG_TPU_MUSE_KERNEL_A1_FAST_NORM": "1"},
        ),
    ):
        hlo = (
            _make_sm(True)
            .trace(
                jax.ShapeDtypeStruct((tp_d * tp_d, c_k, hidden), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((tp_d * hidden, qg_cols), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((tp_d, qg_cols), jnp.float32),
                jax.ShapeDtypeStruct((tp_d * tp_d * c_k, 128), jnp.float32),
                jax.ShapeDtypeStruct((tp_d * tp_d * c_k, 128), jnp.float32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert "_rpal" in hlo and "partition_id" not in hlo


def test_ccs_fuse_ag_a1_mode3_in_kernel_lowering():
    """Verify CCS_FUSE_AG_A1=3 lowers clean in-kernel overlapped 1-hop XOR=1 AG on TP=8 and TP=16."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1

    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip("Requires >=8 JAX devices")

    def _run_lower(tp_d: int, h_q: int, h_kv: int, c_k: int, c_next: int, tag: str):
        mesh = Mesh(np.array(devs[:tp_d]), ("model",))
        qg_cols = 2 * h_q * 64 + h_kv * 128

        def _body(ag_k, next_sl, wqkvg, wsc, cos_k, sin_k):
            q_k, g_k, kv_k, ag_n = kernel_a1.pallas_a1_chunk_1way(
                ag_k,
                wqkvg,
                wsc,
                cos_k.reshape(tp_d, c_k, 128),
                sin_k.reshape(tp_d, c_k, 128),
                None,
                t_shard=c_k,
                row_off=0,
                h_q_local=h_q,
                h_kv_local=h_kv,
                rpa_layout=True,
                mode="all",
                tp_axis="model",
                tm=256 if c_k % 256 == 0 else 128,
                next_ag_in=next_sl,
            )
            return (
                q_k.reshape(1, tp_d * c_k, h_q // 2, 2, 128),
                g_k.reshape(tp_d * c_k, h_q * 64),
                kv_k.reshape(tp_d * c_k, 1, 4, 128),
                ag_n,
            )

        sm = jax.jit(
            jax.shard_map(
                _body,
                mesh=mesh,
                in_specs=(P("model"),) * 6,
                out_specs=(P("model"),) * 4,
                check_vma=False,
            )
        )
        shlo = (
            sm.trace(
                jax.ShapeDtypeStruct((tp_d * tp_d, c_k, 8192), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((tp_d * c_next, 8192), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((tp_d * 8192, qg_cols), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((tp_d, qg_cols), jnp.float32),
                jax.ShapeDtypeStruct((tp_d * tp_d * c_k, 128), jnp.float32),
                jax.ShapeDtypeStruct((tp_d * tp_d * c_k, 128), jnp.float32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert tag in shlo and "partition_id" not in shlo

    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
    ):
        _run_lower(8, 16, 2, 512, 512, "_inag_xy")
        if len(devs) >= 16:
            _run_lower(16, 8, 1, 512, 512, "_inag_xyz")
            _run_lower(16, 8, 1, 512, 1280, "_inag_xyz")

            # 16-device Mosaic interpret=True + detect_races=True 0-ULP parity on c_next=128 (64,32,32).
            from jax import lax
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu

            orig_pc = pl.pallas_call

            def _mosaic_interp(*args, **kwargs):
                kwargs["interpret"] = pltpu.InterpretParams(
                    detect_races=True, uninitialized_memory="nan"
                )
                return orig_pc(*args, **kwargs)

            mesh16 = Mesh(np.array(devs[:16]), ("model",))
            tp_d, c_k, c_next, h_q, h_kv, hidden = 16, 128, 128, 8, 1, 256
            qg_cols = 2 * h_q * 64 + h_kv * 128
            rng = np.random.default_rng(99)
            ag_k = jnp.asarray(
                (rng.standard_normal((tp_d * tp_d, c_k, hidden)) * 0.2).astype(np.float32)
            ).astype(jnp.float8_e4m3fn)
            next_sl = jnp.asarray(
                (rng.standard_normal((tp_d * c_next, hidden)) * 0.2).astype(np.float32)
            ).astype(jnp.float8_e4m3fn)
            wqkvg = jnp.asarray(
                (rng.standard_normal((tp_d * hidden, qg_cols)) * 0.05).astype(np.float32)
            ).astype(jnp.float8_e4m3fn)
            wsc = jnp.asarray(rng.uniform(0.01, 0.05, size=(tp_d, qg_cols)).astype(np.float32))
            cos_k = jnp.asarray(np.ones((tp_d * tp_d * c_k, 128), dtype=np.float32))
            sin_k = jnp.asarray(np.zeros((tp_d * tp_d * c_k, 128), dtype=np.float32))

            def _fused(ag_in, nxt, w, ws, c, s):
                q_k, g_k, kv_k, ag_n = kernel_a1.pallas_a1_chunk_1way(
                    ag_in, w, ws, c.reshape(tp_d, c_k, 128), s.reshape(tp_d, c_k, 128),
                    None, t_shard=c_k, row_off=0, h_q_local=h_q, h_kv_local=h_kv,
                    rpa_layout=False, mode="all", tp_axis="model", tm=128, next_ag_in=nxt,
                )
                return q_k.reshape(tp_d * c_k, h_q * 64), g_k.reshape(tp_d * c_k, h_q * 64), kv_k.reshape(tp_d * c_k, 128), ag_n

            def _ref(ag_in, nxt, w, ws, c, s):
                ag_n = lax.all_gather(nxt, "model", axis=0, tiled=False)
                q_k, g_k, kv_k = kernel_a1.pallas_a1_chunk_1way(
                    ag_in, w, ws, c.reshape(tp_d, c_k, 128), s.reshape(tp_d, c_k, 128),
                    None, t_shard=c_k, row_off=0, h_q_local=h_q, h_kv_local=h_kv,
                    rpa_layout=False, mode="all", tp_axis="model", tm=128,
                )
                return q_k.reshape(tp_d * c_k, h_q * 64), g_k.reshape(tp_d * c_k, h_q * 64), kv_k.reshape(tp_d * c_k, 128), ag_n

            with mock.patch.object(pl, "pallas_call", _mosaic_interp):
                qf, gf, kvf, agnf = jax.jit(
                    jax.shard_map(_fused, mesh=mesh16, in_specs=(P("model"),) * 6, out_specs=(P("model"),) * 4, check_vma=False)
                )(ag_k, next_sl, wqkvg, wsc, cos_k, sin_k)
                qr, gr, kvr, agnr = jax.jit(
                    jax.shard_map(_ref, mesh=mesh16, in_specs=(P("model"),) * 6, out_specs=(P("model"),) * 4, check_vma=False)
                )(ag_k, next_sl, wqkvg, wsc, cos_k, sin_k)

            np.testing.assert_array_equal(np.asarray(qf.view(jnp.uint16)), np.asarray(qr.view(jnp.uint16)))
            np.testing.assert_array_equal(np.asarray(gf.view(jnp.uint16)), np.asarray(gr.view(jnp.uint16)))
            np.testing.assert_array_equal(np.asarray(kvf.view(jnp.uint8)), np.asarray(kvr.view(jnp.uint8)))
            np.testing.assert_array_equal(np.asarray(agnf.view(jnp.uint8)), np.asarray(agnr.view(jnp.uint8)))


def test_ep16_sortfree_routing_parity_and_compact_sort_guard():
    """Verify EP=16 PALLAS_SORTFREE_ROUTING=2 parity at T=4096 and _sort_local_packed_lhs_indices guard."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _sort_local_packed_lhs_indices,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.routing_pallas import (
        pallas_global_local_pos,
        pallas_local_inv_perm_and_counts,
    )

    devs = jax.devices()
    if len(devs) < 16:
        pytest.skip("Requires 16 JAX devices")

    mesh = Mesh(np.array(devs[:16]), ("model",))
    rng = np.random.default_rng(123)
    t_total, t_loc = 4096, 256
    g_ids_np = np.stack(
        [rng.choice(256, size=8, replace=False) for _ in range(t_total)], axis=0
    ).astype(np.int32)
    g_ids = jnp.asarray(g_ids_np)
    go_arr = jax.device_put(
        jnp.arange(16, dtype=jnp.int32) * 16, NamedSharding(mesh, P("model"))
    )
    me_arr = jax.device_put(
        jnp.arange(16, dtype=jnp.int32), NamedSharding(mesh, P("model"))
    )
    l_ids = jax.device_put(
        g_ids_np.reshape(16, t_loc, 8), NamedSharding(mesh, P("model", None, None))
    )

    def _sf_eval(li, gi, go, me):
        lp, gs, gsa, ltr = pallas_local_inv_perm_and_counts(
            li[0], gi, go[0], me[0], ep_size=16, num_local_experts=16, total_id_buckets=256
        )
        glp = pallas_global_local_pos(
            gi, go[0], num_local_experts=16, ep_size=16, rotate_rs=True
        )
        return lp[None], gs[None], gsa[None], ltr[None], glp[None]

    with _interpret_on_cpu():
        _, gs, _, ltr, glp = jax.jit(
            jax.shard_map(
                _sf_eval,
                mesh=mesh,
                in_specs=(P("model", None, None), P(), P("model"), P("model")),
                out_specs=(
                    P("model", None, None),
                    P("model", None),
                    P("model", None),
                    P("model"),
                    P("model", None, None),
                ),
                check_vma=False,
            )
        )(l_ids, g_ids, go_arr, me_arr)

    for r in range(16):
        mask_r = (g_ids_np >= r * 16) & (g_ids_np < (r + 1) * 16)
        own_hits = int(mask_r[r * t_loc : (r + 1) * t_loc].sum())
        assert int(gs[r, r * 16 : (r + 1) * 16].sum()) == int(mask_r.sum())
        assert int(ltr[r]) == t_loc * 8 - own_hits
        valid_pos = np.asarray(glp[r])[mask_r]
        assert np.array_equal(np.sort(valid_pos), np.arange(int(mask_r.sum())))

    # Verify compact_mode=1 safely falls back on T=4096 (n_flat=32768 < 65536).
    tok_ids_flat = jnp.arange(t_total * 8, dtype=jnp.int32) >> 3
    lhs = _sort_local_packed_lhs_indices(
        global_topk_ids=g_ids,
        topk_indices_flat=g_ids.reshape(-1),
        tok_ids_flat=tok_ids_flat,
        go_val=jnp.int32(0),
        my_id=jnp.int32(0),
        num_tokens=t_total,
        chunk_size=t_loc,
        ep_size=16,
        num_local_experts=16,
        topk=8,
        rotate_rs=True,
        compact_mode=1,
    )
    assert lhs.shape == (t_total * 8,)

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _allow_pallas_sortfree_routing,
    )

    assert not _allow_pallas_sortfree_routing(
        1,
        sortfree_prefill=True,
        topk=8,
        ep_size=16,
        t_local=256,
        n_loc_eff=18,
        has_pad_sentinel=True,
        eff_hoist_routing=True,
        eff_cm_pipe=True,
    )
    assert not _allow_pallas_sortfree_routing(
        1,
        sortfree_prefill=True,
        topk=8,
        ep_size=16,
        t_local=256,
        n_loc_eff=18,
        has_pad_sentinel=False,
        eff_hoist_routing=True,
        eff_cm_pipe=True,
    )
    assert _allow_pallas_sortfree_routing(
        1,
        sortfree_prefill=True,
        topk=8,
        ep_size=8,
        t_local=512,
        n_loc_eff=34,
        has_pad_sentinel=True,
        eff_hoist_routing=True,
        eff_cm_pipe=True,
    )

    # Verify ROUTING_NATIVE_PACK=1 remains supported at tl=256 in standalone routing_pallas.
    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
        mock.patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_ROUTING_NATIVE_PACK": "1"}),
    ):
        sm_tl256 = jax.jit(
            jax.shard_map(
                lambda li, gi, go, me: pallas_local_inv_perm_and_counts(
                    li[0], gi, go[0], me[0], ep_size=16, num_local_experts=18,
                    total_id_buckets=289, has_pad_sentinel=True,
                )[0][None],
                mesh=mesh,
                in_specs=(P("model", None, None), P(), P("model"), P("model")),
                out_specs=P("model", None, None),
                check_vma=False,
            )
        )
        hlo_tl256 = (
            sm_tl256.trace(
                jax.ShapeDtypeStruct((16, 256, 8), jnp.int32),
                jax.ShapeDtypeStruct((4096, 8), jnp.int32),
                jax.ShapeDtypeStruct((16,), jnp.int32),
                jax.ShapeDtypeStruct((16,), jnp.int32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert "pallas_moe_sortfree_route_np_tl256" in hlo_tl256


def test_tp16_a1_rpa_layout_integration_gating_and_inag_cache():
    """Verify MusePrefillKernels.from_model allows A1_RPA_LAYOUT=1 on native TP16 v6 and separates _inag cache."""
    import types

    import torch
    from sglang_tpu.muse_prefill_kernels import integration

    cfg = types.SimpleNamespace(
        num_attention_heads=128,
        num_key_value_heads=16,
        qk_scale_factor=12.0 ** -0.5 * 1.4426950408889634 if False else 0.0,
        output_multiplier=0.17677669529663687,
        routed_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        hidden_size=8192,
        moe_hidden_size=4096,
        num_local_experts=256,
        num_experts_per_tok=8,
        head_dim=64,
    )
    from sglang_tpu.muse_prefill_kernels.attn_kernels.reference import QUERY_SCALE

    cfg.qk_scale_factor = QUERY_SCALE
    parallel = types.SimpleNamespace(
        tp_size=16, attn_tp_size=16, moe_ep_size=16, attn_dcp_size=1
    )
    fake_decoder = types.SimpleNamespace(
        pp_group=types.SimpleNamespace(world_size=1),
        layers=[],
    )
    fake_model = types.SimpleNamespace(model=fake_decoder, text_config=cfg)

    with (
        mock.patch("sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False),
        mock.patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
    ):
        # Reject A1_VARIANT='v1' when A1_RPA_LAYOUT=1 on native TP=16.
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_TP16_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_A1_VARIANT": "v1",
                "SGLANG_TPU_MUSE_KERNEL_A3_VARIANT": "v16",
                "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
            },
        ):
            assert integration.MusePrefillKernels.from_model(fake_model) is None

        # Allow A1_VARIANT='v6' + A1_RPA_LAYOUT=1 on native TP=16 (reaches runs_on_kernels/empty-layers -> None).
        fake_decoder.layers = [
            types.SimpleNamespace(
                mlp=types.SimpleNamespace(
                    experts=types.SimpleNamespace(w13_weight=torch.empty(0))
                )
            )
        ]
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_TP16_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_A1_VARIANT": "v6",
                "SGLANG_TPU_MUSE_KERNEL_A3_VARIANT": "v16",
                "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
                "SGLANG_TPU_MUSE_KERNEL_FORCE": "0",
            },
        ):
            # Passes from_model validation and reaches layer inspection cleanly without returning early at reason guard.
            with mock.patch.object(integration.logger, "warning") as mock_warn:
                integration.MusePrefillKernels.from_model(fake_model)
                reasons = [str(call.args[1]) for call in mock_warn.call_args_list if len(call.args) > 1]
                assert not any("native TP16" in r for r in reasons), reasons


def test_build_chunk_rpa_metas_torch_preunbinds_swa_group_pages():
    """Verify build_chunk_rpa_metas_torch pre-unbinds swa_group_pages tuple for zero-view lookup."""
    import types

    import torch
    from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
        build_chunk_rpa_metas_torch,
    )

    md = types.SimpleNamespace(
        kv_lens=torch.tensor([1024, 512], dtype=torch.int32),
        cu_q_lens=torch.tensor([0, 1024, 2048], dtype=torch.int32),
        cu_kv_lens=torch.tensor([0, 1024, 2048], dtype=torch.int32),
        page_indices=torch.arange(16, dtype=torch.int32),
        swa_page_indices=torch.arange(8, dtype=torch.int32),
        swa_group_page_indices=torch.arange(24, dtype=torch.int32).view(3, 8),
    )
    metas = build_chunk_rpa_metas_torch(md, tp_size=8, chunks=(128, 128))
    assert len(metas) == 2
    for m in metas:
        assert m.swa_group_pages is not None and len(m.swa_group_pages) == 3
        for g in range(3):
            torch.testing.assert_close(m.swa_group_pages[g], m.swa_group_page_indices[g])


def test_extend_full_vocab_logits_mode_gating_and_hygiene():
    """Verify SGLANG_TPU_EXTEND_FULL_VOCAB_LOGITS defaults off, uses .get(), and pins EXTEND to FULL_VOCAB."""
    import types

    from sglang.srt.model_executor.forward_batch_info import ForwardMode, LogitsOutputMode
    from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
    from sglang_tpu.sampler import (
        _can_use_fused_gumbel_max_for_batch,
        distributed_top_k_for_batch,
    )

    assert tpu_envs.SGLANG_TPU_EXTEND_FULL_VOCAB_LOGITS.get() is False
    repo = pathlib.Path(__file__).resolve().parents[5]
    for rel in (
        "sglang_tpu/sampler.py",
        "sglang_tpu/compilation/decoder_compile_policy.py",
    ):
        tree = ast.parse((repo / rel).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "SGLANG_TPU_EXTEND_FULL_VOCAB_LOGITS"
            ):
                found_get = any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "get"
                    and c.func.value is node
                    for c in ast.walk(tree)
                )
                assert found_get, f"{rel}:{node.lineno} missing .get()"

    info = types.SimpleNamespace(
        is_all_greedy=False,
        uniform_top_k=50,
        is_any_greedy=False,
        sampling_seed=None,
        need_top_k_sampling=True,
        need_top_p_sampling=False,
        need_min_p_sampling=False,
        has_custom_logit_processor=False,
        logit_bias=None,
        grammars=(),
        sampling_mask_batch_indices=None,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        penalizer_orchestrator=None,
        temperatures=[1.0],
    )
    ext_batch = types.SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=64,
        return_logprob=False,
        sampling_info=info,
        spec_info=None,
    )
    dec_batch = types.SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=64,
        return_logprob=False,
        sampling_info=info,
        spec_info=None,
    )
    with mock.patch.dict(os.environ, {"SGLANG_TPU_EXTEND_FULL_VOCAB_LOGITS": "1"}):
        assert distributed_top_k_for_batch(ext_batch, sampling_backend="tpu") is None
        assert distributed_top_k_for_batch(dec_batch, sampling_backend="tpu") == 50
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_FUSED_GUMBEL_MAX_LM_HEAD": "1",
                "SGLANG_TPU_MUSE_FUSED_FP8_LM_HEAD_TOPK": "1",
            },
        ):
            assert _can_use_fused_gumbel_max_for_batch(ext_batch, None) is False

    pol = DecoderTpuCompilePolicy.__new__(DecoderTpuCompilePolicy)
    pol.model_runner = types.SimpleNamespace(spec_algorithm=None, model=object())
    pol._distributed_top_k_width = lambda: 64
    with mock.patch.dict(os.environ, {"SGLANG_TPU_EXTEND_FULL_VOCAB_LOGITS": "1"}):
        assert pol._extend_logits_output_modes(64) == (LogitsOutputMode.FULL_VOCAB,)
        assert LogitsOutputMode.DISTRIBUTED_TOP_K in pol._warmup_logits_output_modes(64)


def test_d_with_next_in_gather_split_keeps_sparsecore_ag_unless_opted_in():
    """Split _d_with_next_in_gather must keep SparseCore all_gather unless XOR1_AG_UNDER_D=1."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _d_with_next_in_gather,
    )

    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip("Requires >=8 JAX devices")

    mesh8 = Mesh(np.array(devs[:8]), ("model",))

    def _fake_d(x, *, num_rows=None, row_start=0, stream_out=None):
        total = x.shape[0]
        end = total if num_rows is None else row_start + num_rows
        sl = x[row_start:end]
        stream = sl if stream_out is None else jnp.concatenate([stream_out, sl], axis=0)
        return stream, sl

    def _run_hlo(split: bool, env_dict: dict[str, str]) -> str:
        with (
            mock.patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
            ),
            tpu_info_for_cpu_export(),
            mock.patch.dict(os.environ, env_dict),
        ):
            fn = jax.jit(
                jax.shard_map(
                    lambda x: _d_with_next_in_gather(
                        d_fn=_fake_d,
                        d_args=(x[0],),
                        rows0=512,
                        split=split,
                        axis_name="model",
                        gather_static_fp8=False,
                    )[2][None],
                    mesh=mesh8,
                    in_specs=(P("model", None, None),),
                    out_specs=P("model", None, None, None),
                    check_vma=False,
                )
            )
            return (
                fn.trace(jax.ShapeDtypeStruct((8, 1024, 128), jnp.float8_e4m3fn))
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )

    hlo_split_default = _run_hlo(
        True, {"SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "3"}
    )
    assert "all_gather" in hlo_split_default and "xor1_" not in hlo_split_default

    hlo_split_optin = _run_hlo(
        True,
        {
            "SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "3",
            "SGLANG_TPU_MUSE_KERNEL_XOR1_AG_UNDER_D": "1",
        },
    )
    assert "xor1_xytorus_ag_tp8" in hlo_split_optin

    hlo_unsplit = _run_hlo(
        False, {"SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "3"}
    )
    assert "xor1_xytorus_ag_tp8" in hlo_unsplit

    # XOR1_BRIDGE_AG=4 (MoE-only) and XOR1_AG_MOE=3 keep both split and unsplit D on SparseCore.
    for env_d in (
        {"SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "4"},
        {"SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "0", "SGLANG_TPU_MUSE_KERNEL_XOR1_AG_MOE": "3"},
    ):
        for sp in (True, False):
            hlo_sc = _run_hlo(sp, env_d)
            assert "all_gather" in hlo_sc and "xor1_" not in hlo_sc


def test_xor1_callsite_split_moe_and_chunk_b2_lowering():
    """Verify XOR1_BRIDGE_AG=4 and XOR1_AG_{MOE,CHUNK,B2} route cleanly per call site."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _moe_b2_all_gather,
    )

    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip("Requires >=8 JAX devices")
    mesh8 = Mesh(np.array(devs[:8]), ("model",))

    def _trace_moe_b2(env_dict: dict[str, str]) -> str:
        with (
            mock.patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
            ),
            tpu_info_for_cpu_export(),
            mock.patch.dict(os.environ, env_dict),
        ):
            fn = jax.jit(
                jax.shard_map(
                    lambda x: _moe_b2_all_gather(x[0], axis_name="model")[None],
                    mesh=mesh8,
                    in_specs=(P("model", None, None),),
                    out_specs=P("model", None, None, None),
                    check_vma=False,
                )
            )
            return (
                fn.trace(jax.ShapeDtypeStruct((8, 1024, 4224), jnp.int8))
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )

    hlo_off = _trace_moe_b2({"SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "0"})
    assert "all_gather" in hlo_off and "xor1_" not in hlo_off

    hlo_xag4 = _trace_moe_b2({"SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "4"})
    assert "xor1_xytorus_ag_tp8" in hlo_xag4 and "partition_id" not in hlo_xag4

    hlo_moe3 = _trace_moe_b2(
        {
            "SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": "0",
            "SGLANG_TPU_MUSE_KERNEL_XOR1_AG_MOE": "3",
        }
    )
    assert "xor1_xytorus_ag_tp8" in hlo_moe3 and "partition_id" not in hlo_moe3


def test_tp16_ccs_fuse_ag_a1_nonzero_raises_loudly():
    """Verify #2833 §1: TP=16 rejects CCS_FUSE_AG_A1 != 0 loudly in from_model and _ops."""
    import types

    from sglang.srt.configs.muse_spark import MuseSparkTextConfig
    from sglang_tpu.muse_prefill_kernels import integration
    from sglang_tpu.muse_prefill_kernels.attn_kernels.reference import QUERY_SCALE

    cfg = MuseSparkTextConfig(
        hidden_size=8192,
        moe_intermediate_size=4096,
        num_local_experts=256,
        num_experts_per_tok=8,
        num_attention_heads=128,
        num_key_value_heads=16,
        head_dim=64,
    )
    cfg.qk_scale_factor = QUERY_SCALE
    parallel = types.SimpleNamespace(
        tp_size=16, attn_tp_size=16, moe_ep_size=16, attn_dcp_size=1
    )
    fake_model = types.SimpleNamespace(
        model=types.SimpleNamespace(
            pp_group=types.SimpleNamespace(world_size=1), layers=[]
        ),
        text_config=cfg,
    )

    for bad_val in ("1", "2", "3"):
        with (
            mock.patch.dict(
                os.environ, {"SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1": bad_val}
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ),
        ):
            with pytest.raises(ValueError, match="CCS_FUSE_AG_A1 != 0"):
                integration.MusePrefillKernels.from_model(fake_model)

    fake_mesh = types.SimpleNamespace(shape={"model": 16})
    with mock.patch(
        "sglang_tpu.moe.pallas.fused_moe_rs_op._get_mesh", return_value=fake_mesh
    ):
        with pytest.raises(ValueError, match="CCS_FUSE_AG_A1 != 0"):
            integration._ops(
                h_q_local=8,
                h_kv_local=1,
                a1_variant="v6",
                a3_variant="v16",
                fused_glue=False,
                head_norm_eps=1e-8,
                ccs_fuse_ag_a1=1,
            )


def test_xor1_nan_poisoned_scratch_interpret_parity():
    """Verify TP=8 xy and TP=16 flat/xyz xor1 AG kernels have 0 NaN leaks under NaN-poisoned scratch."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from sglang_tpu.kernels.xor1_bridge_all_gather import xor1_bridge_all_gather

    devs = jax.devices()
    if len(devs) < 8:
        pytest.skip("Requires >=8 JAX devices")

    orig_pc = pl.pallas_call

    def _nan_interp(*args, **kwargs):
        kwargs["interpret"] = pltpu.InterpretParams(
            detect_races=True, uninitialized_memory="nan"
        )
        return orig_pc(*args, **kwargs)

    cases = [(8, "3", 128)]
    if len(devs) >= 16:
        cases.extend([(16, "2", 128), (16, "3", 128)])

    rng = np.random.default_rng(2770)
    for tp_d, mode_val, rows in cases:
        mesh = Mesh(np.array(devs[:tp_d]), ("model",))
        x_np = (rng.standard_normal((tp_d, rows, 128)) * 0.5).astype(np.float32)
        x_j = jax.device_put(
            jnp.asarray(x_np, dtype=jnp.bfloat16),
            NamedSharding(mesh, P("model", None, None)),
        )
        with (
            mock.patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
            ),
            tpu_info_for_cpu_export(),
            mock.patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_XOR1_BRIDGE_AG": mode_val}),
            mock.patch.object(pl, "pallas_call", _nan_interp),
        ):
            out = jax.jit(
                jax.shard_map(
                    lambda s: xor1_bridge_all_gather(
                        s[0], axis_name="model", tiled=False
                    )[None],
                    mesh=mesh,
                    in_specs=(P("model", None, None),),
                    out_specs=P("model", None, None, None),
                    check_vma=False,
                )
            )(x_j)
        out_u16 = np.asarray(out.view(jnp.uint16))
        ref_u16 = np.asarray(x_j.view(jnp.uint16))
        assert not np.any(np.isnan(np.asarray(out, dtype=np.float32)))
        for r in range(tp_d):
            np.testing.assert_array_equal(out_u16[r], ref_u16)


def test_muse_nan_trap_optin_stage_and_pad_isolation():
    """Verify SGLANG_TPU_MUSE_NAN_TRAP is default-OFF (0-cost), ignores pad NaNs, and traps real-token NaNs."""
    import types

    import torch

    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.compilation.direct_compile import _collect_callbacks
    from sglang_tpu.muse_prefill_kernels import integration

    assert tpu_envs.SGLANG_TPU_MUSE_NAN_TRAP.get() is False

    tp_size, t_shard = 16, 8
    t_total = tp_size * t_shard
    chunks = (4, 4)
    real_tokens = 99  # straddles chunks and ranks in chunk-major order

    pk = integration.MusePrefillKernels.__new__(integration.MusePrefillKernels)
    torch.nn.Module.__init__(pk)
    pk.start = 0
    pk.tp_size = tp_size
    pk.tp_rank = 12  # owns chunk0 [48..52) (all real) and chunk1 [112..116) (all pad)
    pk._cm_perm_cache = {}
    pk.nan_trap = False

    fb = types.SimpleNamespace(
        extend_seq_lens=torch.tensor([real_tokens], dtype=torch.int32)
    )
    stream = torch.zeros((t_total, 4), dtype=torch.float32)
    stream_shard = torch.zeros((t_shard, 4), dtype=torch.float32)

    assert (
        pk._init_nan_trap(
            forward_batch=fb,
            stream=stream,
            stream_shard=stream_shard,
            chunks=chunks,
            t_shard=t_shard,
            t_total=t_total,
        )
        is None
    )

    pk.nan_trap = True
    ctx = pk._init_nan_trap(
        forward_batch=fb,
        stream=stream,
        stream_shard=stream_shard,
        chunks=chunks,
        t_shard=t_shard,
        t_total=t_total,
    )
    assert ctx is not None
    # Rank 12 in chunks=(4,4): local rows 0..3 -> orig 48..51 (real), rows 4..7 -> orig 112..115 (pad).
    assert ctx["shard_orig"].tolist() == [48, 49, 50, 51, 112, 113, 114, 115]

    # Pad-only NaNs (including fp8_e4m3fn 0x7F in KV, bf16 NaN in Q [H, T, D], shard pad rows) must NOT fire.
    q_pad = torch.zeros((2, t_total, 4), dtype=torch.bfloat16)
    q_pad[:, real_tokens:, :] = float("nan")
    kv_pad_u8 = torch.zeros((t_total, 4), dtype=torch.uint8)
    kv_pad_u8[real_tokens:, :] = 0x7F
    kv_pad = kv_pad_u8.view(torch.float8_e4m3fn)
    shard_pad = torch.zeros((t_shard, 4), dtype=torch.bfloat16)
    shard_pad[4:, :] = float("nan")  # orig tokens 112..115 >= 99
    integration._record_nan_trap(
        ctx, 7, 1, ((q_pad, 1), (kv_pad, 0)), ctx["full_real"], ctx["orig_full"]
    )
    integration._record_nan_trap(
        ctx, 7, 4, ((shard_pad, 0),), ctx["shard_real"], ctx["shard_orig"]
    )
    pk._emit_nan_trap(ctx)
    assert pk.last_nan_trap == []

    # Real-token NaNs: inject at orig token 98 in RPA (stage 2) and orig token 50 in MoE (stage 4).
    rpa_bad = torch.zeros((t_total, 4), dtype=torch.bfloat16)
    rpa_bad[98, 2] = float("nan")
    shard_bad = torch.zeros((t_shard, 4), dtype=torch.bfloat16)
    shard_bad[2, 1] = float("inf")  # shard_orig[2] == 50
    ctx2 = pk._init_nan_trap(
        forward_batch=fb,
        stream=stream,
        stream_shard=stream_shard,
        chunks=chunks,
        t_shard=t_shard,
        t_total=t_total,
    )
    integration._record_nan_trap(
        ctx2, 19, 2, ((rpa_bad, 0),), ctx2["full_real"], ctx2["orig_full"]
    )
    integration._record_nan_trap(
        ctx2, 19, 4, ((shard_bad, 0),), ctx2["shard_real"], ctx2["shard_orig"]
    )

    # Verify insert_callback registration + _make_nan_trap_callback binding + error logging
    # even while torch.compiler.is_compiling() is True (as direct_compile sets it during trace).
    prev_compiling = torch.compiler._is_compiling_flag
    torch.compiler._is_compiling_flag = True
    try:
        with _collect_callbacks() as cbs:
            pk._emit_nan_trap(ctx2)
    finally:
        torch.compiler._is_compiling_flag = prev_compiling
    assert len(cbs) == 1
    assert cbs[0].callback_factory is integration._make_nan_trap_callback
    fake_model = types.SimpleNamespace(prefill_kernels=pk)
    bound_cb = cbs[0].callback_factory(fake_model)
    with mock.patch.object(integration.logger, "error") as err_log:
        bound_cb(cbs[0].tensor)
    assert pk.last_nan_trap == [[19, 2, 1, 98], [19, 4, 1, 50]]
    err_log.assert_called_once()
    assert "MUSE_NAN_TRAP" in err_log.call_args.args[0]

    # Verify torch.compile path (_is_compiling_flag=True without _collect_callbacks):
    # buffer latch + CompiledForwardShim._flush_prefill_nan_trap -> flush_nan_trap_host().
    from sglang_tpu.compilation.compiled_forward_shim import CompiledForwardShim

    pk.nan_trap_first = torch.full((4,), -1, dtype=torch.int32)
    pk.nan_trap_last = torch.full((4,), -1, dtype=torch.int32)
    pk.nan_trap_count = torch.zeros(1, dtype=torch.int32)
    pk._nan_trap_seen = 0
    pk._nan_trap_armed_logged = False
    pk.last_nan_trap = []

    prev_compiling = torch.compiler._is_compiling_flag
    torch.compiler._is_compiling_flag = True
    try:
        pk._emit_nan_trap(ctx)  # pad-only: count stays 0
        pk._emit_nan_trap(ctx2)  # real NaN: latches [19, 2, 1, 98] and [19, 4, 1, 50]
    finally:
        torch.compiler._is_compiling_flag = prev_compiling

    assert int(pk.nan_trap_count[0]) == 2
    shim = CompiledForwardShim.__new__(CompiledForwardShim)
    shim.fn = lambda *a, **kw: None
    shim.policy = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(
            model=types.SimpleNamespace(model=types.SimpleNamespace(prefill_kernels=pk))
        )
    )
    ext_fb = types.SimpleNamespace(
        forward_mode=types.SimpleNamespace(is_extend=lambda: True)
    )
    with mock.patch.object(integration.logger, "error") as err_log:
        shim._flush_prefill_nan_trap(shim.fn, ext_fb)
        # Subsequent clean batch does not duplicate error log.
        shim._flush_prefill_nan_trap(shim.fn, ext_fb)
    assert pk.last_nan_trap == [[19, 2, 1, 98], [19, 4, 1, 50]]
    err_log.assert_called_once()
    assert "MUSE_NAN_TRAP" in err_log.call_args.args[0]

    # Real TorchDynamo torch.compile(fullgraph=True) trace of _init_nan_trap + _record_nan_trap + _emit_nan_trap:
    # proves insert_callback does not trip Dynamo on _ACTIVE_CALLBACKS.get().
    pk.nan_trap_first.fill_(-1)
    pk.nan_trap_last.fill_(-1)
    pk.nan_trap_count.zero_()
    pk._nan_trap_seen = 0
    pk.last_nan_trap = []

    def _compiled_trap_step(s_in, ss_in, rpa_in, fb_in):
        c_dyn = pk._init_nan_trap(
            forward_batch=fb_in,
            stream=s_in,
            stream_shard=ss_in,
            chunks=chunks,
            t_shard=t_shard,
            t_total=t_total,
        )
        integration._record_nan_trap(
            c_dyn, 12, 2, ((rpa_in, 0),), c_dyn["full_real"], c_dyn["orig_full"]
        )
        pk._emit_nan_trap(c_dyn)
        return s_in + 0.0

    compiled_step = torch.compile(_compiled_trap_step, backend="eager", fullgraph=True)
    rpa_dyn_bad = torch.zeros((t_total, 4), dtype=torch.bfloat16)
    rpa_dyn_bad[77, 0] = float("nan")
    rpa_dyn_bad[real_tokens:, :] = float("nan")
    fb_dyn = types.SimpleNamespace(
        extend_seq_lens=torch.tensor([real_tokens], dtype=torch.int32)
    )
    compiled_step(stream, stream_shard, rpa_dyn_bad, fb_dyn)
    assert int(pk.nan_trap_count[0]) == 1
    with mock.patch.object(integration.logger, "error") as err_log_dyn:
        shim._flush_prefill_nan_trap(shim.fn, ext_fb)
    assert pk.last_nan_trap == [[12, 2, 1, 77]]
    err_log_dyn.assert_called_once()








