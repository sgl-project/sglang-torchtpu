# SPDX-License-Identifier: Apache-2.0
"""Unit, CPU interpret=True numerical parity, and TPU7x StableHLO verification tests for Wave-8 Prefill Glue + A3 + D + B1/B2."""

from __future__ import annotations

import ast
import contextlib
import functools
import os
import pathlib
from unittest import mock

import numpy as np
import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
    build_chunk_rpa_metas_torch,
    resolve_causal_chunks,
    slice_chunk_rpa_metadata_jax,
    to_chunk_major_perm,
)
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

_W8_FLAGS = (
    "SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS",
    "SGLANG_TPU_MUSE_KERNEL_CCS_CHUNK_A3_PROLOGUE",
    "SGLANG_TPU_MUSE_KERNEL_D_OPT",
    "SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE",
    "SGLANG_TPU_MUSE_KERNEL_B1_B2_PREPACK",
    "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST",
    "SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING",
    "SGLANG_TPU_MUSE_KERNEL_FP8_STREAM_GATHER",
    "SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1",
    "SGLANG_TPU_MUSE_KERNEL_OVERLAP_DEQ0_ROUTING_AG",
)


@contextlib.contextmanager
def _interpret_on_cpu():
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    real = pl.pallas_call

    def interpret(*args, **kwargs):
        kwargs["interpret"] = pltpu.InterpretParams()
        return real(*args, **kwargs)

    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
        mock.patch.object(pl, "pallas_call", interpret),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        yield


def test_w8_prefill_glue_flags_default_off_and_dot_get():
    """All Wave-8 prefill glue flags must default strictly off and always use .get()."""
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_STAGE_SENDS.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_CCS_CHUNK_A3_PROLOGUE.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_OPT.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_PREPACK.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_FP8_STREAM_GATHER.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_CCS_FUSE_AG_A1.get() == 0
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_OVERLAP_DEQ0_ROUTING_AG.get() is False

    repo = pathlib.Path(__file__).resolve().parents[5]
    checked = set(_W8_FLAGS)
    rel_paths = (
        "sglang_tpu/muse_prefill_kernels/integration.py",
        "sglang_tpu/muse_prefill_kernels/attn_kernels/kernel_a3.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/kernel_b1.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/kernel_b2.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/kernel_d.py",
        "sglang_tpu/muse_prefill_kernels/moe_bridge_kernels/pipeline.py",
    )
    for rel in rel_paths:
        src = (repo / rel).read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in checked:
                # Either .get() directly or passed to getattr() then .get()
                has_get = any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "get"
                    and c.func.value is node
                    for c in ast.walk(tree)
                )
                assert has_get, f"{rel}:{node.lineno} accesses {node.attr} without .get()"


def test_a3_v16_parity_stagger_1factorization_and_jax_match():
    """Verify A3 v16 parity-staggered peer schedule is a symmetric 1-factorization on 2x2x1."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from sglang_tpu.muse_prefill_kernels.attn_kernels.kernel_a3 import (
        _a3_v16_step_peer_jax,
        a3_v16_step_peer,
    )

    step_xors = tuple(range(8))
    for parity_stagger in (False, True):
        for me in range(8):
            peers = [
                a3_v16_step_peer(me, d, 8, parity_stagger=parity_stagger)
                for d in range(8)
            ]
            # Bijection over all 8 ranks
            assert sorted(peers) == list(range(8))
            assert peers[0] == me
            assert peers[1] == (me ^ 1)
            for d, p in enumerate(peers):
                # Symmetric involution: peer(peer(me, d), d) == me -> d_recv == d_send == d!
                assert a3_v16_step_peer(p, d, 8, parity_stagger=parity_stagger) == me
                p_jax = int(
                    _a3_v16_step_peer_jax(
                        jnp.int32(me),
                        d,
                        8,
                        step_xors=step_xors,
                        parity_stagger=parity_stagger,
                    )
                )
                assert p_jax == p

    # Verify orthogonal 2x2 torus link usage at d=2..5: the two TCs of every chip {2c, 2c+1}
    # communicate with two DIFFERENT neighbor chips (1 horizontal + 1 vertical, 0 collisions).
    for d in (2, 3, 4, 5):
        for chip in range(4):
            p0 = a3_v16_step_peer(2 * chip, d, 8, parity_stagger=True) // 2
            p1 = a3_v16_step_peer(2 * chip + 1, d, 8, parity_stagger=True) // 2
            assert p0 != p1, f"Same-chip TCs collided on neighbor chip {p0} at d={d}"
            assert p0 != chip and p1 != chip


def test_a3_stage_sends_modes_hlo_and_rpa_out_prologue():
    """Verify A3 _opt_wo_kernel lowers cleanly for A3_STAGE_SENDS=0..3 and RPA-out chunk prologue."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_prefill_kernels.attn_kernels.kernel_a3 import (
        pallas_gate_static_fp8_quant,
        pallas_option3d_zero_copy9_persrc_async_a2a_wo_3d_static_fp8,
    )

    tp_deg = 8 if len(jax.devices()) >= 8 else 1
    mesh = Mesh(np.array(jax.devices()[:tp_deg]), ("model",))

    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
    ):
        for mode in (0, 1, 2, 3):
            shmapped = jax.shard_map(
                lambda q, w, s, m=mode: pallas_option3d_zero_copy9_persrc_async_a2a_wo_3d_static_fp8(
                    q,
                    w,
                    s,
                    tp_axis="model",
                    opt_wo=True,
                    stage_sends_mode=m,
                ),
                mesh=mesh,
                in_specs=(P(None, None, None), P(None, None, None), P(None, None)),
                out_specs=P("model", None),
                check_vma=False,
            )
            fn = jax.jit(shmapped)
            # Trace across 2048 rows per shard (num_t_half=2) and full tp_deg (8 on 8-dev mesh, 1 on 1-dev)
            lowered = fn.trace(
                jax.ShapeDtypeStruct((tp_deg, 2048, 1024), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((tp_deg, 1024, 8192), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            ).lower(lowering_platforms=("tpu",))
            hlo = lowered.as_text("stablehlo")
            assert "custom_call" in hlo

        # Verify pallas_gate_static_fp8_quant lowers identically for 5D [1, c_tok, Hq//2, 2, 128]
        # and 3D [c_tok, Hq, 128] under rpa_packed_out=True (including tm=64 for c_tok=64 chunks)
        fn_5d = jax.jit(
            lambda raw, g: pallas_gate_static_fp8_quant(
                raw, g, tm=128, head_norm_eps=1e-5, rpa_packed_out=True
            )
        )
        fn_3d = jax.jit(
            lambda raw, g: pallas_gate_static_fp8_quant(
                raw, g, tm=128, head_norm_eps=1e-5, rpa_packed_out=True
            )
        )
        fn_tm64 = jax.jit(
            lambda raw, g: pallas_gate_static_fp8_quant(
                raw, g, tm=64, head_norm_eps=1e-5, rpa_packed_out=True
            )
        )
        hlo_5d = (
            fn_5d.trace(
                jax.ShapeDtypeStruct((1, 256, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((256, 1024), jnp.bfloat16),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        hlo_3d = (
            fn_3d.trace(
                jax.ShapeDtypeStruct((256, 16, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((256, 1024), jnp.bfloat16),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        hlo_64 = (
            fn_tm64.trace(
                jax.ShapeDtypeStruct((1, 64, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((64, 1024), jnp.bfloat16),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert "pallas_gate_static_fp8_quant_rpa_out" in hlo_5d
        assert "pallas_gate_static_fp8_quant_rpa_out" in hlo_3d
        assert "pallas_gate_static_fp8_quant_rpa_out" in hlo_64

        # Also verify tm=64 vs tm=128 CPU interpret parity on standard 2D normed/raw layout
        rng = np.random.default_rng(123)
        raw_2d = rng.standard_normal((128, 1024)).astype(np.float32) * 0.5
        gate_2d = rng.standard_normal((128, 1024)).astype(np.float32) * 0.5
        with _interpret_on_cpu():
            q_tm64 = np.asarray(
                pallas_gate_static_fp8_quant(
                    jnp.asarray(raw_2d, dtype=jnp.bfloat16),
                    jnp.asarray(gate_2d, dtype=jnp.bfloat16),
                    tm=64,
                    head_norm_eps=1e-5,
                    rpa_packed_out=False,
                ).astype(jnp.float32)
            )
            q_tm128 = np.asarray(
                pallas_gate_static_fp8_quant(
                    jnp.asarray(raw_2d, dtype=jnp.bfloat16),
                    jnp.asarray(gate_2d, dtype=jnp.bfloat16),
                    tm=128,
                    head_norm_eps=1e-5,
                    rpa_packed_out=False,
                ).astype(jnp.float32)
            )
            np.testing.assert_array_equal(q_tm64, q_tm128)


def test_sortfree_chunk_rpa_metadata_zero_sort_and_exact_parity():
    """Verify sort-free chunk RPA metadata matches argsort bit-for-bit across random masks and emits 0 sorts."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(2026)
    for bs in (1, 2, 4, 8, 16, 64):
        pps = 32
        for trial in range(5):
            active_prob = [0.0, 1.0, 0.5, 0.25, 0.75][trial]
            q_lens = [
                int(rng.integers(128, 2048)) if rng.random() < active_prob else 0
                for _ in range(bs)
            ]
            prefixes = [int(rng.integers(0, 1024)) for _ in range(bs)]
            kv_lens_np = np.array(
                [p + q if q > 0 else 0 for p, q in zip(prefixes, q_lens)],
                dtype=np.int32,
            )
            cu_q_np = np.concatenate([[0], np.cumsum(q_lens, dtype=np.int32)]).astype(
                np.int32
            )
            cu_kv_np = (np.arange(bs + 1, dtype=np.int32) * (pps * 128)).astype(
                np.int32
            )
            page_np = rng.integers(0, 4096, size=(bs * pps,), dtype=np.int32)
            swa_np = rng.integers(0, 4096, size=(bs * pps,), dtype=np.int32)

            class _Md:
                pass

            md = _Md()
            md.kv_lens = torch.from_numpy(kv_lens_np)
            md.cu_q_lens = torch.from_numpy(cu_q_np)
            md.cu_kv_lens = torch.from_numpy(cu_kv_np)
            md.page_indices = torch.from_numpy(page_np)
            md.swa_page_indices = torch.from_numpy(swa_np)
            md.swa_group_page_indices = None

            chunks = (64, 128, 64)
            metas = build_chunk_rpa_metas_torch(md, tp_size=8, chunks=chunks)
            t_start = 0
            for k, c_k in enumerate(chunks):
                t_end = t_start + 8 * c_k
                q0 = np.clip(cu_q_np[:-1], t_start, t_end)
                q1 = np.clip(cu_q_np[1:], t_start, t_end)
                ql = np.maximum(0, q1 - q0)
                act = ql > 0
                rem = np.maximum(0, cu_q_np[1:] - q1)
                kvl = np.where(act, np.maximum(0, kv_lens_np - rem), 0)
                ref_ord = np.argsort(np.where(act, 0, 1), kind="stable")

                assert metas[k].kv_lens.dtype == torch.int32
                assert metas[k].cu_q_lens.dtype == torch.int32
                assert metas[k].cu_kv_lens.dtype == torch.int32
                assert metas[k].page_indices.dtype == torch.int32
                assert metas[k].swa_page_indices.dtype == torch.int32
                assert metas[k].distribution.dtype == torch.int32
                np.testing.assert_array_equal(metas[k].kv_lens.numpy(), kvl[ref_ord])
                np.testing.assert_array_equal(
                    np.diff(metas[k].cu_q_lens.numpy()), ql[ref_ord]
                )
                np.testing.assert_array_equal(
                    metas[k].page_indices.numpy(),
                    page_np.reshape(bs, pps)[ref_ord].reshape(-1),
                )
                np.testing.assert_array_equal(
                    metas[k].swa_page_indices.numpy(),
                    swa_np.reshape(bs, pps)[ref_ord].reshape(-1),
                )

                kv_j, cuq_j, cukv_j, pg_j, dist_j = slice_chunk_rpa_metadata_jax(
                    jnp.asarray(kv_lens_np),
                    jnp.asarray(cu_q_np),
                    jnp.asarray(cu_kv_np),
                    jnp.asarray(page_np),
                    t_start=t_start,
                    t_end=t_end,
                )
                assert kv_j.dtype == jnp.int32
                assert cuq_j.dtype == jnp.int32
                assert cukv_j.dtype == jnp.int32
                assert pg_j.dtype == jnp.int32
                assert dist_j.dtype == jnp.int32
                np.testing.assert_array_equal(np.asarray(kv_j), kvl[ref_ord])
                np.testing.assert_array_equal(np.diff(np.asarray(cuq_j)), ql[ref_ord])
                np.testing.assert_array_equal(
                    np.asarray(pg_j), page_np.reshape(bs, pps)[ref_ord].reshape(-1)
                )
                assert int(dist_j[2]) == int(act.sum())
                t_start = t_end

    # Verify lowered StableHLO of slice_chunk_rpa_metadata_jax contains ZERO sort ops!
    lowered = jax.jit(
        lambda kvl, cuq, cukv, pg: slice_chunk_rpa_metadata_jax(
            kvl, cuq, cukv, pg, t_start=0, t_end=4096
        )
    ).trace(
        jax.ShapeDtypeStruct((8,), jnp.int32),
        jax.ShapeDtypeStruct((9,), jnp.int32),
        jax.ShapeDtypeStruct((9,), jnp.int32),
        jax.ShapeDtypeStruct((8 * 128,), jnp.int32),
    ).lower()
    assert "stablehlo.sort" not in lowered.as_text("stablehlo")


def test_kernel_d_opt_and_mxu_combine_interpret_parity():
    """Verify Kernel D D_OPT=1/3 and D_MXU_COMBINE=1 match baseline numerics on CPU interpret=True."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_d import (
        pallas_combine_unpool_residual_nextnorm_4slab,
    )

    chunk_size = 128
    keys = jax.random.split(jax.random.key(99), 7)
    out_buf_3d = (
        jax.random.normal(keys[0], (chunk_size * 8, 32, 128), jnp.float32) * 0.4
    ).astype(jnp.bfloat16)
    inv_perm_c = jax.random.randint(
        keys[1], (chunk_size, 8), 0, chunk_size * 8, jnp.int32
    )
    raw_w = jax.random.uniform(keys[2], (chunk_size, 8), jnp.float32, minval=0.05, maxval=1.0)
    topk_w = (raw_w / jnp.sum(raw_w, axis=-1, keepdims=True)).astype(jnp.bfloat16)
    stream_mid = jax.random.normal(keys[3], (chunk_size, 8192), jnp.float32) * 0.5
    wunpool = (jax.random.normal(keys[4], (4096, 8192), jnp.float32) * 0.1).astype(
        jnp.float8_e4m3fn
    )
    wunpool_sc = jnp.full((8192,), 0.02, jnp.float32)
    post_ffn_w = jax.random.normal(keys[5], (8192,), jnp.float32) * 0.1
    alpha_ffn = jnp.full((8192,), 0.9, jnp.float32)
    beta_ffn = jnp.full((8192,), 1.1, jnp.float32)
    next_in_w = jax.random.normal(keys[6], (8192,), jnp.float32) * 0.1

    args = (
        out_buf_3d,
        inv_perm_c,
        topk_w,
        stream_mid,
        wunpool,
        wunpool_sc,
        post_ffn_w,
        alpha_ffn,
        beta_ffn,
        next_in_w,
    )

    with _interpret_on_cpu():
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_D_T64_C16": "1",
                "SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP": "1",
                "SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL": "4096",
                "SGLANG_TPU_MUSE_KERNEL_D_OPT": "0",
                "SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE": "0",
            },
        ):
            ref_s, ref_ni = pallas_combine_unpool_residual_nextnorm_4slab(
                *args, tok_tile=64, emit_static_fp8_attn_in=True
            )

        # D_OPT=1: bit-exact DMA + finer _so + 256-col Phase 3/4
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_D_T64_C16": "1",
                "SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP": "1",
                "SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL": "4096",
                "SGLANG_TPU_MUSE_KERNEL_D_OPT": "1",
                "SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE": "0",
            },
        ):
            opt1_s, opt1_ni = pallas_combine_unpool_residual_nextnorm_4slab(
                *args, tok_tile=64, emit_static_fp8_attn_in=True
            )
        np.testing.assert_allclose(
            np.asarray(opt1_s), np.asarray(ref_s), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(
            np.asarray(opt1_ni.astype(jnp.float32)),
            np.asarray(ref_ni.astype(jnp.float32)),
        )

        # D_OPT=3 + D_MXU_COMBINE=1 (including split row_start=64!)
        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_D_T64_C16": "1",
                "SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP": "1",
                "SGLANG_TPU_MUSE_KERNEL_D_FP8_UNPOOL": "4096",
                "SGLANG_TPU_MUSE_KERNEL_D_OPT": "3",
                "SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE": "1",
            },
        ):
            s0, ni0 = pallas_combine_unpool_residual_nextnorm_4slab(
                *args, tok_tile=64, num_rows=64, emit_static_fp8_attn_in=True
            )
            s_all, ni1 = pallas_combine_unpool_residual_nextnorm_4slab(
                *args,
                tok_tile=64,
                row_start=64,
                stream_out=s0,
                emit_static_fp8_attn_in=True,
            )
            ni_all = jnp.concatenate([ni0, ni1], axis=0)

        s_ref_np = np.asarray(ref_s)
        s_got_np = np.asarray(s_all)
        cos_s = float(
            np.sum(s_ref_np * s_got_np)
            / (np.linalg.norm(s_ref_np) * np.linalg.norm(s_got_np))
        )
        assert cos_s > 0.99999

        ni_ref_np = np.asarray(ref_ni.astype(jnp.float32))
        ni_got_np = np.asarray(ni_all.astype(jnp.float32))
        cos_ni = float(
            np.sum(ni_ref_np * ni_got_np)
            / (np.linalg.norm(ni_ref_np) * np.linalg.norm(ni_got_np))
        )
        assert cos_ni > 0.9995


def test_b1_b2_fast_sub_strips_prepack_and_skip_sc_dma():
    """Verify B1/B2 256-lane sub-strips, prepacked statics, and skipped dead sc DMA match reference."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_b1 import (
        pallas_postattn_res_prenorm_router_topk_fp8,
        prepare_b1_static_inputs,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_b2 import (
        pallas_fused_b1_b2_wpool_quant,
        pallas_wpool_prenorm_int8_quant_fp8_nstream,
        prepare_b2_static_inputs,
    )

    t_rows = 256
    keys = jax.random.split(jax.random.key(314), 8)
    attn_s = (jax.random.normal(keys[0], (t_rows, 8192), jnp.float32) * 0.5).astype(
        jnp.bfloat16
    )
    stream_s = jax.random.normal(keys[1], (t_rows, 8192), jnp.float32) * 0.5
    alpha_a = jnp.full((8192,), 0.95, jnp.float32)
    beta_a = jnp.full((8192,), 1.05, jnp.float32)
    pffn_w = (jax.random.normal(keys[2], (8192,), jnp.float32) * 0.1).astype(
        jnp.bfloat16
    )
    wrouter_bf16 = (jax.random.normal(keys[3], (8192, 256), jnp.float32) * 0.05).astype(
        jnp.bfloat16
    )
    wrouter_hilo_pre = jnp.concatenate(
        [wrouter_bf16, jnp.zeros_like(wrouter_bf16)], axis=0
    )
    bias = jax.random.normal(keys[4], (256,), jnp.float32) * 0.01
    wpool = (jax.random.normal(keys[5], (8192, 4096), jnp.float32) * 0.1).astype(
        jnp.float8_e4m3fn
    )
    wpool_sc = jnp.full((4096,), 0.02, jnp.float32)
    pe_w = (jax.random.normal(keys[6], (4096,), jnp.float32) * 0.1).astype(jnp.bfloat16)

    b1_vecs, b1_bias_t, _, _ = prepare_b1_static_inputs(
        alpha_a, beta_a, pffn_w, bias, wrouter_hilo_pre, router_mode="bf16_hilo_pre"
    )
    b2_vecs = prepare_b2_static_inputs(wpool_sc, pe_w)

    # Verify pack_sc_in_i8=True in pallas_wpool_prenorm_int8_quant_fp8_nstream has 1 output (no dead sc DMA!)
    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
    ):
        for sw in ("0", "1"):
            with mock.patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_MUSE_KERNEL_B2_SINGLE_WRITE": sw,
                    "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST": "1",
                },
            ):
                lowered_b2 = jax.jit(
                    lambda rf, rsc, wp, bv: pallas_wpool_prenorm_int8_quant_fp8_nstream(
                        rf, rsc, wp, bv, pe_w, pack_sc_in_i8=True
                    )[0]
                ).trace(
                    jax.ShapeDtypeStruct((256, 8192), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((256, 128), jnp.float32),
                    jax.ShapeDtypeStruct((8192, 4096), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8, 4096), jnp.float32),
                ).lower(lowering_platforms=("tpu",))
                hlo_b2 = lowered_b2.as_text("stablehlo")
                assert "tensor<256x128xbf16>" not in hlo_b2

    with _interpret_on_cpu():
        with mock.patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST": "0"}):
            ref_r, ref_sc, ref_smid, ref_ids, ref_ws = (
                pallas_postattn_res_prenorm_router_topk_fp8(
                    attn_s,
                    stream_s,
                    alpha_a,
                    beta_a,
                    pffn_w,
                    wrouter_hilo_pre,
                    bias,
                    router_mode="bf16_hilo_pre",
                )
            )
            ref_i8, ref_b2_sc = pallas_wpool_prenorm_int8_quant_fp8_nstream(
                ref_r, ref_sc, wpool, wpool_sc, pe_w, pack_sc_in_i8=True
            )
            assert ref_b2_sc is None
            ref_fb_i8, ref_fb_smid, ref_fb_ids, ref_fb_ws = (
                pallas_fused_b1_b2_wpool_quant(
                    attn_s,
                    stream_s,
                    alpha_a,
                    beta_a,
                    pffn_w,
                    wrouter_hilo_pre,
                    bias,
                    wpool,
                    wpool_sc,
                    pe_w,
                    router_mode="bf16_hilo_pre",
                    fp8_wpool=True,
                    half_idx=0,
                )
            )

        with mock.patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST": "1"}):
            got_r, got_sc, got_smid, got_ids, got_ws = (
                pallas_postattn_res_prenorm_router_topk_fp8(
                    attn_s,
                    stream_s,
                    b1_vecs,
                    beta_a,
                    pffn_w,
                    wrouter_hilo_pre,
                    b1_bias_t,
                    router_mode="bf16_hilo_pre",
                )
            )
            got_i8, got_b2_sc = pallas_wpool_prenorm_int8_quant_fp8_nstream(
                got_r, got_sc, wpool, b2_vecs, pe_w, pack_sc_in_i8=True
            )
            assert got_b2_sc is None
            got_fb_i8, got_fb_smid, got_fb_ids, got_fb_ws = (
                pallas_fused_b1_b2_wpool_quant(
                    attn_s,
                    stream_s,
                    b1_vecs,
                    beta_a,
                    pffn_w,
                    wrouter_hilo_pre,
                    b1_bias_t,
                    wpool,
                    b2_vecs,
                    pe_w,
                    router_mode="bf16_hilo_pre",
                    fp8_wpool=True,
                    half_idx=0,
                )
            )

        np.testing.assert_allclose(
            np.asarray(got_smid), np.asarray(ref_smid), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(np.asarray(got_ids), np.asarray(ref_ids))
        np.testing.assert_allclose(
            np.asarray(got_ws.astype(jnp.float32)),
            np.asarray(ref_ws.astype(jnp.float32)),
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            np.asarray(got_i8[:, :4096].astype(jnp.float32)),
            np.asarray(ref_i8[:, :4096].astype(jnp.float32)),
            rtol=0,
            atol=1.0,
        )
        np.testing.assert_allclose(
            np.asarray(got_fb_smid), np.asarray(ref_fb_smid), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(np.asarray(got_fb_ids), np.asarray(ref_fb_ids))
        np.testing.assert_allclose(
            np.asarray(got_fb_i8[:, :4096].astype(jnp.float32)),
            np.asarray(ref_fb_i8[:, :4096].astype(jnp.float32)),
            rtol=0,
            atol=1.0,
        )


def test_zero_sort_block_scatter_routing_zero_sorts_and_duplicate_safe():
    """Verify ZERO_SORT_ROUTING=1 emits 0 stablehlo.sort ops and handles duplicate row buckets."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        compute_block_scatter_routing,
    )

    rng = np.random.default_rng(2027)
    num_tokens = 512
    ep_size = 8
    num_local = 33
    # Include deliberate duplicate bucket hits within the same token row + pad sentinels!
    g_ids_np = rng.integers(0, ep_size * num_local, size=(num_tokens, 8), dtype=np.int32)
    g_ids_np[0, :4] = 5  # 4 duplicates of local expert 5 on rank 0!
    g_ids_np[-64:, :] = ep_size * num_local  # pad sentinel rows

    with mock.patch.dict(
        os.environ,
        {
            "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
            "SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING": "0",
            "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
        },
    ):
        ref_lhs, ref_gs, ref_gstart, ref_trecv = compute_block_scatter_routing(
            jnp.asarray(g_ids_np),
            jnp.int32(0),
            jnp.int32(0),
            ep_size=ep_size,
            num_local_experts=num_local,
            topk=8,
            has_pad_sentinel=True,
        )

    with mock.patch.dict(
        os.environ,
        {
            "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
            "SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING": "1",
            "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
        },
    ):
        fn = jax.jit(
            lambda g: compute_block_scatter_routing(
                g,
                jnp.int32(0),
                jnp.int32(0),
                ep_size=ep_size,
                num_local_experts=num_local,
                topk=8,
                has_pad_sentinel=True,
            )
        )
        lowered = fn.trace(
            jax.ShapeDtypeStruct((num_tokens, 8), jnp.int32)
        ).lower()
        assert "stablehlo.sort" not in lowered.as_text("stablehlo")
        got_lhs, got_gs, got_gstart, got_trecv = fn(jnp.asarray(g_ids_np))

    tot_local = int(np.sum(np.asarray(ref_gs)))
    np.testing.assert_array_equal(np.asarray(got_gs), np.asarray(ref_gs))
    np.testing.assert_array_equal(np.asarray(got_gstart), np.asarray(ref_gstart))
    assert int(got_trecv) == int(ref_trecv)
    np.testing.assert_array_equal(
        np.asarray(got_lhs)[:tot_local], np.asarray(ref_lhs)[:tot_local]
    )


def test_fp8_stream_gather_and_chunk_major_unpermute_parity():
    """Verify FP8 pre-AG stream quantization + fused 1-byte chunk-major unpermute accuracy."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax import lax

    from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
        permute_to_chunk_major_jax,
        resolve_causal_chunks,
        unpermute_from_chunk_major_jax,
    )

    tp = 8
    t_shard = 512
    t_total = tp * t_shard
    chunks = resolve_causal_chunks(t_shard)
    rng = np.random.default_rng(77)
    stream_orig = jnp.asarray(
        rng.standard_normal((t_total, 8192)).astype(np.float32) * 2.5
    )
    stream_cm = permute_to_chunk_major_jax(stream_orig, tp, chunks)

    # Simulate per-rank shard -> FP8 + packed bf16 scale -> AG -> 1-byte CM unpermute -> dequant
    shards_packed = []
    for r in range(tp):
        x = stream_cm[r * t_shard : (r + 1) * t_shard]
        row_amax = jnp.max(jnp.abs(x), axis=-1, keepdims=True)
        row_sc = jnp.where(
            row_amax == 0,
            jnp.bfloat16(0.0),
            (row_amax * jnp.float32(1.0 / 448.0)).astype(jnp.bfloat16),
        )
        sc_f32 = row_sc.astype(jnp.float32)
        inv_sc = jnp.where(sc_f32 == 0, 0.0, 1.0 / jnp.maximum(sc_f32, 1e-30))
        q_u8 = lax.bitcast_convert_type(
            jnp.clip(x * inv_sc, -448.0, 448.0).astype(jnp.float8_e4m3fn), jnp.uint8
        )
        sc_u16 = lax.bitcast_convert_type(row_sc, jnp.uint16).astype(jnp.uint32)
        lo_u8 = (sc_u16 & jnp.uint32(0xFF)).astype(jnp.uint8)
        hi_u8 = lax.shift_right_logical(sc_u16, jnp.uint32(8)).astype(jnp.uint8)
        lane = jnp.arange(128, dtype=jnp.int32)[None, :]
        sc_u8 = jnp.where(
            lane < 64,
            jnp.broadcast_to(lo_u8, (t_shard, 128)),
            jnp.broadcast_to(hi_u8, (t_shard, 128)),
        )
        shards_packed.append(jnp.concatenate([q_u8, sc_u8], axis=1))

    ag_u8 = jnp.concatenate(shards_packed, axis=0)
    ag_u8_unperm = unpermute_from_chunk_major_jax(ag_u8, tp, chunks)
    q_full = lax.bitcast_convert_type(ag_u8_unperm[:, :8192], jnp.float8_e4m3fn)
    ps = ag_u8_unperm[:, 8192:]
    lo16 = ps[:, 0:1].astype(jnp.uint32)
    hi16 = lax.shift_left(ps[:, 64:65].astype(jnp.uint32), jnp.uint32(8))
    sc_full = lax.bitcast_convert_type((lo16 | hi16).astype(jnp.uint16), jnp.bfloat16)
    recovered = (q_full.astype(jnp.bfloat16) * sc_full).astype(jnp.float32)

    orig_np = np.asarray(stream_orig)
    rec_np = np.asarray(recovered)
    cos = float(
        np.sum(orig_np * rec_np) / (np.linalg.norm(orig_np) * np.linalg.norm(rec_np))
    )
    assert cos > 0.999


def test_w8_stacked_moe_pipeline_shard_map_export():
    """Verify full Wave-8 stacked MoE pipeline (prepack + B1/B2 fast + 0-sort + D_OPT=3 + D_MXU_COMBINE=1) lowers cleanly."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
        mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
                "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
                "SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE": "1",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                "SGLANG_TPU_MUSE_KERNEL_ZERO_SORT_ROUTING": "1",
                "SGLANG_TPU_MUSE_KERNEL_B1_B2_PREPACK": "1",
                "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST": "1",
                "SGLANG_TPU_MUSE_KERNEL_D_OPT": "3",
                "SGLANG_TPU_MUSE_KERNEL_D_MXU_COMBINE": "1",
            },
        ),
    ):
        in_specs = (
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
        )
        out_specs = (
            P("model", None),
            P("model", None),
            P("model", None),
            P("model", None),
            P(None, None, None),
        )
        shmapped = jax.shard_map(
            lambda a, s, b1v, dummy1, dummy2, wr, eb, wp, b2v, pew, w1, w2, s1, s2, pnw, go, wu, wus, pffnw, affn, bffn, ninw: fused_moe_full_bridge_and_ffn(
                a,
                s,
                b1v,
                dummy1,
                dummy2,
                wr,
                eb,
                wp,
                b2v,
                pew,
                w1,
                w2,
                s1,
                s2,
                pnw,
                go,
                wu,
                wus,
                pffnw,
                affn,
                bffn,
                ninw,
                mesh=mesh,
                router_mode="bf16_hilo_pre",
                b2_layout="nstream",
                b2_two_half=True,
                b2_deq0_first=True,
                hoist_routing=True,
                d_layout="4slab",
                d_tok_tile=64,
                in_shard_map=True,
                b2_deq0_new_buffer=True,
                next_in_gather_rows=256,
                next_in_gather_static_fp8=True,
            ),
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )
        fn = jax.jit(shmapped)
        lowered = fn.trace(
            jax.ShapeDtypeStruct((8192, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((8192, 8192), jnp.float32),
            jax.ShapeDtypeStruct((8, 8192), jnp.float32),  # prepacked b1_vecs (8x8192)
            jax.ShapeDtypeStruct((8192,), jnp.float32),
            jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
            jax.ShapeDtypeStruct((16384, 256), jnp.bfloat16),  # prepacked wrouter_hilo [2*H, E]
            jax.ShapeDtypeStruct((256, 128), jnp.float32),  # prepacked b1_bias_t
            jax.ShapeDtypeStruct((8192, 4096), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((8, 4096), jnp.float32),  # prepacked b2_vecs (8x4096)
            jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
            jax.ShapeDtypeStruct((256, 4096, 8192), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((256, 4096, 4096), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((256, 16, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((256, 16, 4096), jnp.bfloat16),
            jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
            jax.ShapeDtypeStruct((8,), jnp.int32),
            jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((8192,), jnp.float32),
            jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
            jax.ShapeDtypeStruct((8192,), jnp.float32),
            jax.ShapeDtypeStruct((8192,), jnp.float32),
            jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
        ).lower(lowering_platforms=("tpu",))
        hlo = lowered.as_text("stablehlo")
        assert "pallas_fused_b1_b2" in hlo
        assert "stablehlo.sort" not in hlo
        assert "partition_id" not in hlo


def test_row_tiled_dequant_gmm_layout_16k_28k_32k_parity_and_hlo():
    """Verify row-tiled dequant_*_gmm_layout bit-exact parity and T=16k/28k/32k VMEM-safe lowering."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax import lax

    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_b2 import (
        dequant_both_halves_gmm_layout,
        dequant_half_gmm_layout,
        dequant_single_gmm_layout,
    )

    rng = np.random.default_rng(2028)
    ep_sz = 2
    half_tok = 384  # 3 tiles of 128 -> exercises multi-tile rt indexing!
    t_loc = 2 * half_tok
    hidden = 4096
    i8_vals = rng.integers(-127, 128, size=(ep_sz, t_loc, hidden), dtype=np.int8)
    sc_f32 = (rng.uniform(0.01, 0.25, size=(ep_sz, t_loc, 1))).astype(np.float32)
    sc_bf16 = jnp.asarray(sc_f32, dtype=jnp.bfloat16)
    sc_u16 = lax.bitcast_convert_type(sc_bf16, jnp.uint16).astype(jnp.int32)
    lo_i8 = np.asarray((sc_u16 & 0xFF).astype(jnp.int8))
    hi_i8 = np.asarray(((sc_u16 >> 8) & 0xFF).astype(jnp.int8))
    ps_np = np.concatenate(
        [np.broadcast_to(lo_i8, (ep_sz, t_loc, 64)), np.broadcast_to(hi_i8, (ep_sz, t_loc, 64))],
        axis=-1,
    )
    packed_np = np.concatenate([i8_vals, ps_np], axis=-1)
    packed_j = jnp.asarray(packed_np, dtype=jnp.int8)
    ref_5d = np.asarray(
        (jnp.asarray(i8_vals, dtype=jnp.bfloat16) * sc_bf16)
        .reshape(ep_sz, t_loc, hidden // 128, 128)
        .astype(jnp.float32)
    )

    with _interpret_on_cpu():
        got_single = np.asarray(dequant_single_gmm_layout(packed_j).astype(jnp.float32))
        np.testing.assert_array_equal(got_single, ref_5d)

        p0 = packed_j[:, :half_tok, :]
        p1 = packed_j[:, half_tok:, :]
        buf0 = dequant_half_gmm_layout(p0, None, None, zero_spill_256=True)
        buf01 = dequant_half_gmm_layout(p1, None, buf0, zero_spill_256=True)
        got_halves = np.asarray(
            buf01.reshape(ep_sz, t_loc, hidden // 128, 128).astype(jnp.float32)
        )
        np.testing.assert_array_equal(got_halves, ref_5d)

        got_both = np.asarray(
            dequant_both_halves_gmm_layout(p0, None, p1, None)
            .reshape(ep_sz, t_loc, hidden // 128, 128)
            .astype(jnp.float32)
        )
        np.testing.assert_array_equal(got_both, ref_5d)

    # Verify TPU7x StableHLO lowering at T=16k (t_loc=2048), T=28k (t_loc=3584), T=32k (t_loc=4096)
    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
    ):
        for t_l in (2048, 3584, 4096):
            ht = t_l // 2
            h_single = (
                jax.jit(dequant_single_gmm_layout)
                .trace(jax.ShapeDtypeStruct((8, t_l, 4224), jnp.int8))
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )
            assert "dequant_single_gmm_layout_psc" in h_single
            assert "partition_id" not in h_single

            h_half = (
                jax.jit(
                    lambda a0, a1: dequant_half_gmm_layout(
                        a1,
                        None,
                        dequant_half_gmm_layout(a0, None, None, zero_spill_256=True),
                        zero_spill_256=True,
                    )
                )
                .trace(
                    jax.ShapeDtypeStruct((8, ht, 4224), jnp.int8),
                    jax.ShapeDtypeStruct((8, ht, 4224), jnp.int8),
                )
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )
            assert "dequant_half0_gmm_layout_psc_zs256" in h_half
            assert "dequant_half1_gmm_layout_psc_zs256" in h_half

            h_both = (
                jax.jit(lambda a0, a1: dequant_both_halves_gmm_layout(a0, None, a1, None))
                .trace(
                    jax.ShapeDtypeStruct((8, ht, 4224), jnp.int8),
                    jax.ShapeDtypeStruct((8, ht, 4224), jnp.int8),
                )
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )
            assert "dequant_both_halves_gmm_layout_psc" in h_both



def test_ccs_fuse_ag_a1_and_overlap_deq0_routing_ag_8dev_hlo_and_parity():
    """Verify fused CCS a1_chunk_agd (modes 1 & 2) bit-exact parity + OVERLAP_DEQ0_ROUTING_AG HLO ordering."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax import lax
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.kernels.xor1_bridge_all_gather import xor1_bridge_all_gather
    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )

    if len(jax.devices()) < 8:
        pytest.skip("Requires >=8 virtual/physical JAX devices")

    mesh = Mesh(np.array(jax.devices()[:8]), ("model",))
    tp_d = 8
    c_k = 128
    c_next = 128
    h_q_local = 16
    h_kv_local = 2
    hidden = 8192
    q_cols = h_q_local * 64
    qg_cols = 2 * q_cols + 2 * h_kv_local * 64

    def _make_fused_and_sep(fuse_mode: int, *, rpa_layout: bool = True):
        use_xor1 = fuse_mode == 2

        def _prep_fp8(x_sl):
            if x_sl.dtype != jnp.float8_e4m3fn:
                return jnp.clip(
                    x_sl.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
                    -448.0,
                    448.0,
                ).astype(jnp.float8_e4m3fn)
            return x_sl

        def _ag(x_f8):
            if use_xor1:
                return xor1_bridge_all_gather(x_f8, axis_name="model", tiled=False)
            return lax.all_gather(x_f8, "model", axis=0, tiled=False)

        def _a1_body(ag_k, wqkvg, wqkvg_sc, cos_k, sin_k):
            w_sc = wqkvg_sc * kernel_a1.STATIC_A1_ACT_SCALE
            q_k, gate_k, kv_k = kernel_a1.pallas_a1_chunk_1way(
                ag_k,
                wqkvg,
                w_sc,
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
            c_tok = tp_d * c_k
            if rpa_layout:
                return (
                    q_k.reshape(1, c_tok, h_q_local // 2, 2, 128),
                    gate_k.reshape(c_tok, h_q_local * 64),
                    kv_k.reshape(c_tok, 1, 4, 128),
                )
            return (
                q_k.reshape(c_tok, h_q_local * 64),
                gate_k.reshape(c_tok, h_q_local * 64),
                kv_k.reshape(c_tok, 2 * h_kv_local * 64),
            )

        def _fused_body(ag_k, next_sl, wqkvg, wqkvg_sc, cos_k, sin_k):
            nf8 = _prep_fp8(next_sl)
            nf8_b, ag_b = lax.optimization_barrier((nf8, ag_k))
            ag_next = _ag(nf8_b)
            q_o, g_o, kv_o = _a1_body(ag_b, wqkvg, wqkvg_sc, cos_k, sin_k)
            return q_o, g_o, kv_o, ag_next

        fused_sm = jax.shard_map(
            _fused_body,
            mesh=mesh,
            in_specs=(P("model"),) * 6,
            out_specs=(P("model"),) * 4,
            check_vma=False,
        )
        ag_sm = jax.shard_map(
            lambda x: _ag(_prep_fp8(x)),
            mesh=mesh,
            in_specs=(P("model"),),
            out_specs=P("model"),
            check_vma=False,
        )
        a1_sm = jax.shard_map(
            _a1_body,
            mesh=mesh,
            in_specs=(P("model"),) * 5,
            out_specs=(P("model"),) * 3,
            check_vma=False,
        )
        return fused_sm, ag_sm, a1_sm

    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
    ):
        for fm in (1, 2):
            fused_sm, _, _ = _make_fused_and_sep(fm)
            for ndtype in (jnp.bfloat16, jnp.float8_e4m3fn):
                hlo = (
                    jax.jit(fused_sm)
                    .trace(
                        jax.ShapeDtypeStruct((tp_d * tp_d, c_k, hidden), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((tp_d * c_next, hidden), ndtype),
                        jax.ShapeDtypeStruct((tp_d * hidden, qg_cols), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((tp_d, qg_cols), jnp.float32),
                        jax.ShapeDtypeStruct((tp_d * tp_d * c_k, 128), jnp.float32),
                        jax.ShapeDtypeStruct((tp_d * tp_d * c_k, 128), jnp.float32),
                    )
                    .lower(lowering_platforms=("tpu",))
                    .as_text("stablehlo")
                )
                assert "pallas_a1_1way_all" in hlo
                assert "partition_id" not in hlo
                if fm == 1:
                    assert "all_gather" in hlo

        rng = np.random.default_rng(2029)
        ag_k_np = rng.standard_normal((tp_d * tp_d, c_k, hidden)).astype(np.float32) * 0.2
        next_np = rng.standard_normal((tp_d * c_next, hidden)).astype(np.float32) * 0.2
        wqg_np = rng.standard_normal((tp_d * hidden, qg_cols)).astype(np.float32) * 0.05
        wsc_np = rng.uniform(0.01, 0.05, size=(tp_d, qg_cols)).astype(np.float32)
        cos_np = np.ones((tp_d * tp_d * c_k, 128), dtype=np.float32)
        sin_np = np.zeros((tp_d * tp_d * c_k, 128), dtype=np.float32)

        fused_sm1, ag_sm1, a1_sm1 = _make_fused_and_sep(1, rpa_layout=False)
        with _interpret_on_cpu():
            ag_k_j = jnp.asarray(ag_k_np).astype(jnp.float8_e4m3fn)
            wqg_j = jnp.asarray(wqg_np).astype(jnp.float8_e4m3fn)
            wsc_j = jnp.asarray(wsc_np)
            cos_j = jnp.asarray(cos_np)
            sin_j = jnp.asarray(sin_np)
            for next_j in (
                jnp.asarray(next_np, dtype=jnp.bfloat16),
                jnp.clip(jnp.asarray(next_np) * 32.0, -448.0, 448.0).astype(
                    jnp.float8_e4m3fn
                ),
            ):
                q_f, g_f, kv_f, agn_f = fused_sm1(
                    ag_k_j, next_j, wqg_j, wsc_j, cos_j, sin_j
                )
                agn_ref = ag_sm1(next_j)
                q_r, g_r, kv_r = a1_sm1(ag_k_j, wqg_j, wsc_j, cos_j, sin_j)
                np.testing.assert_array_equal(
                    np.asarray(q_f.astype(jnp.float32)),
                    np.asarray(q_r.astype(jnp.float32)),
                )
                np.testing.assert_array_equal(
                    np.asarray(g_f.astype(jnp.float32)),
                    np.asarray(g_r.astype(jnp.float32)),
                )
                np.testing.assert_array_equal(
                    np.asarray(kv_f.astype(jnp.float32)),
                    np.asarray(kv_r.astype(jnp.float32)),
                )
                np.testing.assert_array_equal(
                    np.asarray(agn_f.astype(jnp.float32)),
                    np.asarray(agn_ref.astype(jnp.float32)),
                )

        with mock.patch.dict(
            os.environ,
            {
                "SGLANG_TPU_MUSE_KERNEL_FUSED_B1_B2": "4",
                "SGLANG_TPU_MUSE_KERNEL_GMM_LAYOUT_DEQUANT": "1",
                "SGLANG_TPU_MUSE_KERNEL_B2_PACK_BF16_SCALE": "1",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_PREFILL": "1",
                "SGLANG_TPU_MUSE_KERNEL_SORTFREE_SCATTER": "1",
                "SGLANG_TPU_MUSE_KERNEL_B1_B2_PREPACK": "1",
                "SGLANG_TPU_MUSE_KERNEL_B1_B2_FAST": "1",
                "SGLANG_TPU_MUSE_KERNEL_D_OPT": "3",
                "SGLANG_TPU_MUSE_KERNEL_OVERLAP_DEQ0_ROUTING_AG": "1",
            },
        ):
            in_specs = (
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
            )
            out_specs = (
                P("model", None),
                P("model", None),
                P("model", None),
                P("model", None),
                P(None, None, None),
            )
            shmapped = jax.shard_map(
                lambda a, s, b1v, d1, d2, wr, eb, wp, b2v, pew, w1, w2, s1, s2, pnw, go, wu, wus, pffnw, affn, bffn, ninw: fused_moe_full_bridge_and_ffn(
                    a,
                    s,
                    b1v,
                    d1,
                    d2,
                    wr,
                    eb,
                    wp,
                    b2v,
                    pew,
                    w1,
                    w2,
                    s1,
                    s2,
                    pnw,
                    go,
                    wu,
                    wus,
                    pffnw,
                    affn,
                    bffn,
                    ninw,
                    mesh=mesh,
                    router_mode="bf16_hilo_pre",
                    b2_layout="nstream",
                    b2_two_half=True,
                    b2_deq0_first=True,
                    hoist_routing=True,
                    d_layout="4slab",
                    d_tok_tile=64,
                    in_shard_map=True,
                    b2_deq0_new_buffer=True,
                    next_in_gather_rows=256,
                    next_in_gather_static_fp8=True,
                ),
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            )
            hlo_ovl = (
                jax.jit(shmapped)
                .trace(
                    jax.ShapeDtypeStruct((8192, 8192), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8192, 8192), jnp.float32),
                    jax.ShapeDtypeStruct((8, 8192), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((16384, 256), jnp.bfloat16),
                    jax.ShapeDtypeStruct((256, 128), jnp.float32),
                    jax.ShapeDtypeStruct((8192, 4096), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8, 4096), jnp.float32),
                    jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((256, 4096, 8192), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((256, 4096, 4096), jnp.float4_e2m1fn),
                    jax.ShapeDtypeStruct((256, 16, 8192), jnp.bfloat16),
                    jax.ShapeDtypeStruct((256, 16, 4096), jnp.bfloat16),
                    jax.ShapeDtypeStruct((4096,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8,), jnp.int32),
                    jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                )
                .lower(lowering_platforms=("tpu",))
                .as_text("stablehlo")
            )
            assert "partition_id" not in hlo_ovl
            pos_deq0 = hlo_ovl.find("dequant_half0_gmm_layout_psc")
            pos_sort = hlo_ovl.find("stablehlo.sort")
            assert pos_deq0 != -1 and pos_sort != -1
            assert pos_deq0 < pos_sort, "dequant_half0 must precede routing sort!"
