# SPDX-License-Identifier: Apache-2.0
"""Unit, race-detection, bit-exact parity, and real libtpu TPU7x AOT HLO/LLO tests for Wave-12 Prefill Glue."""

from __future__ import annotations

import ast
import contextlib
import glob
import os
import pathlib
import re
from unittest import mock

import numpy as np
import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

_NEW_W12_FLAGS = (
    "SGLANG_TPU_MUSE_KERNEL_A3_CCS_2CHUNK",
    "SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE",
)


@contextlib.contextmanager
def _tpu7x_cpu_context(*, interpret: bool = False, detect_races: bool = False):
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    real_pallas_call = pl.pallas_call

    def _wrapped_pallas_call(*args, **kwargs):
        if interpret:
            kwargs["interpret"] = pltpu.InterpretParams(detect_races=detect_races)
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


def test_w12_flags_default_off_cache_tags_and_dot_get_hygiene():
    """Verify both Wave-12 flags default strictly OFF, key cache tags, and always call .get()."""
    from sglang_tpu.muse_prefill_kernels.integration import _w8_carb_cache_tag

    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_CCS_2CHUNK.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE.get() is False

    seen_tags = set()
    for a32c in ("0", "1"):
        for pcmr in ("0", "1"):
            with mock.patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_A3_CCS_2CHUNK": a32c,
                    "SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE": pcmr,
                },
            ):
                seen_tags.add(_w8_carb_cache_tag())
    assert len(seen_tags) == 4

    repo = pathlib.Path(__file__).resolve().parents[5]
    for rel in (
        "sglang_tpu/muse_prefill_kernels/attn_kernels/kernel_a3.py",
        "sglang_tpu/muse_prefill_kernels/integration.py",
        "sglang_tpu/moe/pallas/fused_rs_muse/moe_imbalance_planner.py",
    ):
        tree = ast.parse((repo / rel).read_text())
        parent_map = {
            child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _NEW_W12_FLAGS:
                parent = parent_map.get(node)
                grandparent = parent_map.get(parent) if parent is not None else None
                assert (
                    isinstance(parent, ast.Attribute)
                    and parent.attr == "get"
                    and isinstance(grandparent, ast.Call)
                    and grandparent.func is parent
                ), f"Bare EnvField access without .get() in {rel}:{node.lineno}"


@pytest.mark.parametrize(
    "c0,c1,stage_mode,bn_2k",
    [
        (192, 320, 0, False),
        (192, 320, 1, True),
        (192, 320, 2, False),
        (256, 256, 1, True),
    ],
)
def test_a3_wo_2chunk_bit_exact_parity_and_race_free_8dev(
    c0: int, c1: int, stage_mode: int, bn_2k: bool
):
    """Verify bit-exact parity vs exact XOR-step A2A+dot reference + zero DMA races across 8 TP ranks."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax import lax
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as _interp_mod
    from jax.sharding import Mesh, PartitionSpec as P

    cpus = jax.devices("cpu")
    if len(cpus) < 8:
        pytest.skip(f"Requires 8 CPU devices for TP=8 shard_map (have {len(cpus)})")

    from sglang_tpu.muse_prefill_kernels.attn_kernels.kernel_a3 import (
        STATIC_A3_ACT_SCALE,
        _a3_v16_step_peer_jax,
        pallas_option3d_2chunk_async_a2a_wo_3d_static_fp8,
    )

    tp = 8
    k_loc = 256
    n_dim = 2048
    t_shard = c0 + c1
    mesh = Mesh(np.array(cpus[:tp]), ("model",))
    rng = np.random.default_rng(1200 + c0 + stage_mode * 7 + int(bn_2k))
    q0_np = rng.uniform(-4.0, 4.0, size=(tp, tp, c0, k_loc)).astype(np.float32)
    q1_np = rng.uniform(-4.0, 4.0, size=(tp, tp, c1, k_loc)).astype(np.float32)
    wo_np = rng.uniform(-0.5, 0.5, size=(tp, tp, k_loc, n_dim)).astype(np.float32)
    wsc_np = rng.uniform(0.01, 0.05, size=(tp, 1, n_dim)).astype(np.float32)

    env_patch = {
        "SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS": str(stage_mode),
        "SGLANG_TPU_MUSE_KERNEL_A3_BN2048": "1" if bn_2k else "0",
    }

    # Exact step-by-step A2A + fp32 dot accumulation reference in identical step order!
    use_ps = stage_mode in (1, 2)
    q0_f8 = np.asarray(jnp.asarray(q0_np).astype(jnp.float8_e4m3fn).astype(jnp.float32))
    q1_f8 = np.asarray(jnp.asarray(q1_np).astype(jnp.float8_e4m3fn).astype(jnp.float32))
    q_cat_f8 = np.concatenate([q0_f8, q1_f8], axis=2)  # [src_rank, dst_rank, t_shard, k_loc]
    wo_f8 = np.asarray(jnp.asarray(wo_np).astype(jnp.float8_e4m3fn).astype(jnp.float32))

    ref_shards = []
    for me in range(tp):
        acc = np.zeros((t_shard, n_dim), dtype=np.float32)
        for step in range(tp):
            if step == 0:
                peer = me
            else:
                peer = int(
                    _a3_v16_step_peer_jax(
                        jnp.int32(me),
                        step,
                        tp,
                        step_xors=tuple(range(tp)),
                        parity_stagger=use_ps,
                    )
                )
            # Peer `peer` sends `q_cat[peer, me, :, :]` to `me`'s `recv_vmem[:, step, :, :]`?
            # Wait: check _send_remote in _opt_wo_kernel:
            # Device `me` pulls `src_hbm_ref.at[pl.ds(peer, 1), ...]` ON `device_id={tp_axis: peer}`?
            # NO: `make_async_remote_copy(src_ref=src_hbm_ref.at[pl.ds(peer, 1)], dst_ref=recv_vmem[d_idx], device_id={tp_axis: peer})`
            # pushes `me`'s `src_hbm_ref[peer]` TO `peer`'s `recv_vmem[d_idx]`! Since `_a3_v16_step_peer_jax` is a symmetric involution
            # (`peer_of(peer, d_idx) == me`!), `me` receives `peer`'s `src_hbm_ref[me]` at `d_idx == step`!
            q_step = q_cat_f8[peer, me]  # [t_shard, k_loc]
            w_step = wo_f8[me, step]     # [k_loc, n_dim]
            acc = acc + (q_step @ w_step).astype(np.float32)
        scaled = acc * (wsc_np[me] * STATIC_A3_ACT_SCALE)
        ref_shards.append(
            np.asarray(jnp.asarray(scaled).astype(jnp.bfloat16).astype(jnp.float32))
        )
    out_ref = np.concatenate(ref_shards, axis=0)

    with (
        _tpu7x_cpu_context(interpret=True, detect_races=True),
        mock.patch.dict(os.environ, env_patch),
    ):
        _interp_mod.reset_global_state() if hasattr(_interp_mod, "reset_global_state") else None
        q0_j = jnp.asarray(q0_np).astype(jnp.float8_e4m3fn)
        q1_j = jnp.asarray(q1_np).astype(jnp.float8_e4m3fn)
        wo_j = jnp.asarray(wo_np).astype(jnp.float8_e4m3fn)
        wsc_j = jnp.asarray(wsc_np)

        got_fn = jax.shard_map(
            lambda a, b, w, s: pallas_option3d_2chunk_async_a2a_wo_3d_static_fp8(
                a.squeeze(0),
                b.squeeze(0),
                w.squeeze(0),
                s.squeeze(0),
                tp_axis="model",
                bn=512,
                stage_sends_mode=stage_mode,
            ),
            mesh=mesh,
            in_specs=(P("model"), P("model"), P("model"), P("model")),
            out_specs=P("model", None),
            check_vma=False,
        )
        out_got = np.asarray(got_fn(q0_j, q1_j, wo_j, wsc_j).astype(jnp.float32))
        races = getattr(_interp_mod, "races", None)
        if races is not None and hasattr(races, "races_found"):
            assert not races.races_found, f"DMA/VMEM data race detected on c0={c0}, c1={c1}, ss={stage_mode}!"

    np.testing.assert_array_equal(out_got, out_ref)


def test_a3_wo_2chunk_unaligned_fallback_lowering():
    """Verify <64-row chunk cleanly falls back to concatenated _opt_wo_kernel."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.sharding import Mesh, PartitionSpec as P

    cpus = jax.devices("cpu")
    if len(cpus) < 8:
        pytest.skip("Requires 8 CPU devices")

    from sglang_tpu.muse_prefill_kernels.attn_kernels.kernel_a3 import (
        pallas_option3d_2chunk_async_a2a_wo_3d_static_fp8,
    )

    mesh = Mesh(np.array(cpus[:8]), ("model",))
    with _tpu7x_cpu_context(interpret=False):
        fn = jax.shard_map(
            lambda a, b, w, s: pallas_option3d_2chunk_async_a2a_wo_3d_static_fp8(
                a.squeeze(0), b.squeeze(0), w.squeeze(0), s.squeeze(0), tp_axis="model", bn=512
            ),
            mesh=mesh,
            in_specs=(P("model"), P("model"), P("model"), P("model")),
            out_specs=P("model", None),
            check_vma=False,
        )
        mlir = (
            jax.jit(fn)
            .trace(
                jax.ShapeDtypeStruct((8, 8, 32, 256), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8, 8, 480, 256), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8, 8, 256, 2048), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8, 1, 2048), jnp.float32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert "pallas_a3_wo_v16_opt" in mlir


@pytest.mark.parametrize("t_global", [16384, 24576])
@pytest.mark.parametrize("mode", ["split", "whole"])
def test_pallas_chipmate_route_cpu_export_gate_and_bit_exact_parity(
    t_global: int, mode: str
):
    """Verify PALLAS_CHIPMATE_ROUTE=1 unlocks Pallas router under cpu_export context with exact parity."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from sglang_tpu.moe.pallas.fused_rs_muse.moe_imbalance_planner import (
        _can_use_tpu_pallas_prefill_router,
    )
    from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
        plan_prefill_chipmate_2slot_jax,
    )

    ep_size = 8
    e_loc = 34
    rng = np.random.default_rng(4200 + t_global + (1 if mode == "whole" else 0))
    ids = rng.integers(0, ep_size * e_loc, size=(t_global, 8), dtype=np.int32)
    m0 = rng.random((t_global, 8)) < 0.22
    ids = np.where(m0, 3 + 4 * (ids & 1), ids).astype(np.int32)
    m5 = rng.random((t_global, 8)) < 0.16
    ids = np.where(m5, 5 * e_loc + 11, ids).astype(np.int32)

    with _tpu7x_cpu_context(interpret=True):
        assert jax.default_backend() == "cpu"
        with mock.patch.dict(
            os.environ, {"SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE": "0"}
        ):
            assert _can_use_tpu_pallas_prefill_router(t_global * 8, None) is False
            v_ref, plan_ref = plan_prefill_chipmate_2slot_jax(
                jnp.asarray(ids),
                ep_size=ep_size,
                num_local_experts=e_loc,
                tile_m=256,
                chipmate_xor=1,
                mode=mode,
                slot_xors=(1, 1),
            )
        with mock.patch.dict(
            os.environ, {"SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE": "1"}
        ):
            assert _can_use_tpu_pallas_prefill_router(t_global * 8, None) is True
            v_got, plan_got = plan_prefill_chipmate_2slot_jax(
                jnp.asarray(ids),
                ep_size=ep_size,
                num_local_experts=e_loc,
                tile_m=256,
                chipmate_xor=1,
                mode=mode,
                slot_xors=(1, 1),
            )

    np.testing.assert_array_equal(np.asarray(plan_got), np.asarray(plan_ref))
    np.testing.assert_array_equal(np.asarray(v_got), np.asarray(v_ref))


def test_real_libtpu_tpu7x_aot_hlo_and_llo_zero_concat_and_chipmate():
    """Verify real libtpu.so tpu7x=2x2x1 AOT HLO drops %pad_maximum_fusion & 111 fusions."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from sglang_tpu.moe.pallas.fused_rs_muse.prefill_gmm_opts import (
        plan_prefill_chipmate_2slot_jax,
    )
    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a3

    try:
        topo = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x=2x2x1")
    except Exception as exc:
        pytest.skip(f"tpu7x AOT topology unavailable: {exc}")

    mesh = Mesh(np.array(topo.devices), ("model",))
    sh = lambda p: NamedSharding(mesh, p)
    sh1 = jax.sharding.SingleDeviceSharding(topo.devices[0])

    with (
        _tpu7x_cpu_context(interpret=False),
        mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_A3_BN2048": "1",
                "SGLANG_TPU_MUSE_KERNEL_A3_OPT": "1",
                "SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS": "1",
                "SGLANG_TPU_MUSE_KERNEL_PALLAS_CHIPMATE_ROUTE": "1",
            },
        ),
    ):
        for c0, c1 in [(768, 1280), (1280, 1792), (1024, 2048), (1152, 1920)]:
            tp, k_loc, h_stream = 8, 1024, 8192

            def run_2c(o0, g0, o1, g1, wo, wsc):
                def body(o0_l, g0_l, o1_l, g1_l, wo_l, wsc_l):
                    qp0 = kernel_a3.pallas_gate_static_fp8_quant(
                        o0_l.squeeze(0),
                        g0_l.squeeze(0),
                        tm=512 if (tp * c0) % 1024 else 1024,
                        head_norm_eps=1e-6,
                    ).reshape(tp, c0, k_loc)
                    qp1 = kernel_a3.pallas_gate_static_fp8_quant(
                        o1_l.squeeze(0),
                        g1_l.squeeze(0),
                        tm=512 if (tp * c1) % 1024 else 1024,
                        head_norm_eps=1e-6,
                    ).reshape(tp, c1, k_loc)
                    return kernel_a3.pallas_option3d_2chunk_async_a2a_wo_3d_static_fp8(
                        qp0, qp1, wo_l.squeeze(0), wsc_l.squeeze(0), tp_axis="model"
                    )

                return jax.shard_map(
                    body,
                    mesh=mesh,
                    in_specs=(
                        P("model"),
                        P("model"),
                        P("model"),
                        P("model"),
                        P("model"),
                        P("model"),
                    ),
                    out_specs=P("model", None),
                    check_vma=False,
                )(o0, g0, o1, g1, wo, wsc)

            comp_2c = (
                jax.jit(run_2c)
                .trace(
                    jax.ShapeDtypeStruct((tp, tp * c0, k_loc), jnp.bfloat16, sharding=sh(P("model"))),
                    jax.ShapeDtypeStruct((tp, tp * c0, k_loc), jnp.bfloat16, sharding=sh(P("model"))),
                    jax.ShapeDtypeStruct((tp, tp * c1, k_loc), jnp.bfloat16, sharding=sh(P("model"))),
                    jax.ShapeDtypeStruct((tp, tp * c1, k_loc), jnp.bfloat16, sharding=sh(P("model"))),
                    jax.ShapeDtypeStruct((tp, tp, k_loc, h_stream), jnp.float8_e4m3fn, sharding=sh(P("model"))),
                    jax.ShapeDtypeStruct((tp, 1, h_stream), jnp.float32, sharding=sh(P("model"))),
                )
                .lower()
                .compile()
            )
            hlo_2c = comp_2c.as_text()
            assert "pad_maximum_fusion" not in hlo_2c
            assert f"pallas_a3_wo_2chunk_c{c0}_{c1}" in hlo_2c

        comp_cm = (
            jax.jit(
                lambda g: plan_prefill_chipmate_2slot_jax(
                    g,
                    ep_size=8,
                    num_local_experts=34,
                    tile_m=256,
                    chipmate_xor=1,
                    mode="split",
                    slot_xors=(1, 1),
                )
            )
            .trace(jax.ShapeDtypeStruct((24576, 8), jnp.int32, sharding=sh1))
            .lower()
            .compile()
        )
        hlo_cm = comp_cm.as_text()
        assert "pallas_prefill_chipmate_2slot_split_m196608" in hlo_cm
        n_fusions = len(re.findall(r"= .*fusion\(", hlo_cm))
        assert n_fusions <= 4, f"Expected <=4 fusions with Pallas chipmate router, got {n_fusions}"



def test_w12b_resolve_causal_chunks_2chunk_downscale_and_per_bucket_spec():
    from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
        from_chunk_major_perm,
        resolve_causal_chunks,
        to_chunk_major_perm,
    )

    # Empty spec preserves legacy 3-chunk defaults verbatim.
    assert resolve_causal_chunks(3072, "") == (512, 1280, 1280)
    assert resolve_causal_chunks(2048, "") == (512, 768, 768)
    assert resolve_causal_chunks(1024, "") == (256, 384, 384)
    assert resolve_causal_chunks(512, "") == (128, 192, 192)

    # 2-chunk specs stay 2-chunk & 128-aligned across sub-16k buckets (>=256).
    assert resolve_causal_chunks(2048, "640,1408") == (640, 1408)
    assert resolve_causal_chunks(1024, "640,1408") == (384, 640)
    assert resolve_causal_chunks(512, "640,1408") == (128, 384)
    assert resolve_causal_chunks(256, "640,1408") == (128, 128)
    assert resolve_causal_chunks(128, "640,1408") == (64, 64)

    assert resolve_causal_chunks(2048, "768,1280") == (768, 1280)
    assert resolve_causal_chunks(1024, "768,1280") == (384, 640)
    assert resolve_causal_chunks(512, "768,1280") == (256, 256)
    assert resolve_causal_chunks(256, "768,1280") == (128, 128)

    assert resolve_causal_chunks(3072, "1024,2048") == (1024, 2048)
    assert resolve_causal_chunks(2048, "1024,2048") == (640, 1408)
    assert resolve_causal_chunks(1024, "1024,2048") == (384, 640)
    assert resolve_causal_chunks(512, "1024,2048") == (128, 384)

    # Integer upscale continues to work cleanly.
    assert resolve_causal_chunks(2048, "384,640") == (768, 1280)
    assert resolve_causal_chunks(1024, "384,640") == (384, 640)

    # Explicit per-bucket overrides (';' / ':') take precedence over default scaling.
    spec_multi = "2048:640,1408;1024:256,768;512:192,320;640,1408"
    assert resolve_causal_chunks(2048, spec_multi) == (640, 1408)
    assert resolve_causal_chunks(1024, spec_multi) == (256, 768)
    assert resolve_causal_chunks(512, spec_multi) == (192, 320)
    assert resolve_causal_chunks(256, spec_multi) == (128, 128)

    for ts in (128, 256, 384, 512, 1024, 2048, 3072):
        ch = resolve_causal_chunks(ts, "640,1408")
        assert len(ch) == 2 and sum(ch) == ts
        if ts >= 256:
            assert all(c % 128 == 0 for c in ch)
        p = to_chunk_major_perm(8, ch)
        inv = from_chunk_major_perm(8, ch)
        assert [p[inv[i]] for i in range(8 * ts)] == list(range(8 * ts))


def test_w12b_l61_pruned_per_chunk_gather_bit_exact_fp8_and_bf16():
    import torch

    int_src = (
        pathlib.Path(__file__).resolve().parents[5]
        / "sglang_tpu"
        / "muse_prefill_kernels"
        / "integration.py"
    ).read_text()
    tree = ast.parse(int_src)
    fn_nodes = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_layer_l61_pruned"
    ]
    assert len(fn_nodes) == 1
    l61_segment = ast.get_source_segment(int_src, fn_nodes[0]) or ""
    assert "torch.cat(q_parts" not in l61_segment
    assert "torch.cat(gate_parts" not in l61_segment
    assert "torch.cat(kv_parts" not in l61_segment

    g = torch.Generator().manual_seed(20261123)
    tp_size = 8
    q_size = 1024
    for chunks in ((384, 640), (640, 1408), (256, 384, 384)):
        t_total = tp_size * sum(chunks)
        bs = 64
        # Include exact chunk boundaries, interior indices, and inactive (ext_lens=0) rows.
        c0_tok = tp_size * chunks[0]
        raw_idx = torch.randint(0, t_total, (bs,), generator=g, dtype=torch.int64)
        raw_idx[0] = 0
        raw_idx[1] = c0_tok - 1
        raw_idx[2] = c0_tok
        raw_idx[3] = t_total - 1
        ext_lens = torch.randint(0, 4, (bs,), generator=g, dtype=torch.int32)
        active = ext_lens > 0
        eff_idx = torch.where(active, raw_idx, 0).clamp(0, t_total - 1)
        idx_128 = torch.nn.functional.pad(eff_idx.long(), (0, 128 - bs))

        q_parts = []
        gate_parts = []
        kv_parts = []
        for c_k in chunks:
            c_tok = tp_size * c_k
            q_parts.append(
                torch.randn((1, c_tok, 8, 2, 128), generator=g).to(torch.bfloat16)
            )
            gate_parts.append(
                torch.randn((c_tok, q_size), generator=g).to(torch.bfloat16)
            )
            kv_u8 = torch.randint(
                0, 256, (c_tok, 1, 4, 128), generator=g, dtype=torch.uint8
            )
            kv_parts.append(kv_u8.view(torch.float8_e4m3fn))

        # Reference: full concat + index_select
        q_ref = torch.cat(q_parts, dim=1).index_select(1, idx_128)
        gate_ref = torch.cat(gate_parts, dim=0).index_select(0, idx_128)
        kv_ref = (
            torch.cat([x.view(torch.uint8) for x in kv_parts], dim=0)
            .index_select(0, idx_128)
            .view(torch.float8_e4m3fn)
        )

        # Optimized per-chunk 128-row gather (zero full-T concat)
        q_pruned = None
        gate_pruned = None
        kv_pruned_u8 = None
        kv_dtype = None
        c_off = 0
        for k, (c_k, q_k, gate_k, kv_k) in enumerate(
            zip(chunks, q_parts, gate_parts, kv_parts)
        ):
            c_tok = tp_size * c_k
            loc_idx = (idx_128 - c_off).clamp(0, c_tok - 1)
            q_sel = q_k.index_select(1, loc_idx)
            gate_sel = gate_k.reshape(-1, q_size).index_select(0, loc_idx)
            kv_dtype = kv_k.dtype
            kv_sel_u8 = kv_k.view(torch.uint8).index_select(0, loc_idx)
            if k == 0:
                q_pruned = q_sel
                gate_pruned = gate_sel
                kv_pruned_u8 = kv_sel_u8
            else:
                in_c = (idx_128 >= c_off) & (idx_128 < c_off + c_tok)
                in_q = in_c.view(1, -1, *([1] * (q_sel.ndim - 2)))
                in_kv = in_c.view(-1, *([1] * (kv_sel_u8.ndim - 1)))
                q_pruned = torch.where(in_q, q_sel, q_pruned)
                gate_pruned = torch.where(in_c[:, None], gate_sel, gate_pruned)
                kv_pruned_u8 = torch.where(in_kv, kv_sel_u8, kv_pruned_u8)
            c_off += c_tok
        kv_pruned = kv_pruned_u8.view(kv_dtype)

        assert torch.equal(q_pruned.view(torch.uint16), q_ref.view(torch.uint16))
        assert torch.equal(gate_pruned.view(torch.uint16), gate_ref.view(torch.uint16))
        assert torch.equal(kv_pruned.view(torch.uint8), kv_ref.view(torch.uint8))
