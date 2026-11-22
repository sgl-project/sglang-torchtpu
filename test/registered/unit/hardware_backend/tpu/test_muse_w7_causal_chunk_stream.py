# SPDX-License-Identifier: Apache-2.0
"""Unit, CPU interpret=True causal parity, and TPU7x Mosaic export tests for Causal Chunked Prefill Streaming."""

from __future__ import annotations

import ast
import contextlib
import functools
import pathlib
from unittest import mock

import numpy as np
import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import (
    build_chunk_rpa_metas_torch,
    causal_chunk_a1_rpa_step_jax,
    chunk_major_global_token_ids_jax,
    from_chunk_major_perm,
    permute_to_chunk_major_jax,
    resolve_causal_chunks,
    slice_chunk_rpa_metadata_jax,
    to_chunk_major_perm,
    unpermute_from_chunk_major_jax,
)
from sglang_tpu.pallas_export import tpu_info_for_cpu_export


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
        mock.patch("sglang_tpu.attention.pallas.util.get_tpu_version", return_value=7),
        mock.patch(
            "sglang_tpu.attention.pallas.ragged_paged_attention_v3.get_tpu_version",
            return_value=7,
        ),
        mock.patch.object(pl, "pallas_call", interpret),
        jax.default_device(jax.devices("cpu")[0]),
    ):
        yield


def test_causal_chunk_stream_flags_default_off():
    """Causal Chunked Prefill Streaming flags must default strictly off."""
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM.get() is False
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_SPEC.get() == ""


def test_causal_chunk_stream_uses_dot_get():
    """Ensure all accesses to CAUSAL_CHUNK_STREAM / SPEC use .get()."""
    repo = pathlib.Path(__file__).resolve().parents[5]
    flags = {
        "SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM",
        "SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_SPEC",
    }
    for rel in ("sglang_tpu/muse_prefill_kernels/integration.py",):
        tree = ast.parse((repo / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in flags:
                assert any(
                    isinstance(c, ast.Call)
                    and isinstance(c.func, ast.Attribute)
                    and c.func.attr == "get"
                    and c.func.value is node
                    for c in ast.walk(tree)
                ), f"{rel}:{node.lineno} accesses {node.attr} without .get()"


def test_resolve_causal_chunks_and_perm_bijection():
    """Verify optimal 3-chunk defaults and chunk-major permutation bijection."""
    assert resolve_causal_chunks(2048) == (512, 768, 768)
    assert resolve_causal_chunks(1024) == (256, 384, 384)
    assert resolve_causal_chunks(512) == (128, 192, 192)
    assert resolve_causal_chunks(2048, "1024,1024") == (1024, 1024)

    for tp, chunks in ((8, (256, 384, 384)), (8, (512, 768, 768)), (4, (64, 128, 64))):
        t_tot = tp * sum(chunks)
        perm = to_chunk_major_perm(tp, chunks)
        inv = from_chunk_major_perm(tp, chunks)
        assert len(perm) == t_tot
        assert sorted(perm) == list(range(t_tot))
        assert [perm[inv[i]] for i in range(t_tot)] == list(range(t_tot))
        assert [inv[perm[i]] for i in range(t_tot)] == list(range(t_tot))

        # Verify each chunk k across tp ranks forms the contiguous global slice [t0..t1).
        t_shard = sum(chunks)
        off = 0
        row_off = 0
        for c_k in chunks:
            gathered = []
            for r in range(tp):
                base = r * t_shard + row_off
                gathered.extend(perm[base : base + c_k])
            assert gathered == list(range(off, off + tp * c_k))
            off += tp * c_k
            row_off += c_k


def test_chunk_rpa_metadata_multi_seq_packed_compaction():
    """Verify compacted per-chunk metadata across multi-seq 2x8k-style packed boundaries."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    torch = pytest.importorskip("torch")

    # 2 active sequences (7500 + 7500 = 15000 tokens, padded to 16384 with pad seq of 1384)
    # Chunk boundaries: (4096, 6144, 6144) -> [0,4096), [4096,10240), [10240,16384).
    bs = 8
    pps = 128
    page_size = 128
    q_lens = [7500, 7500, 1384, 0, 0, 0, 0, 0]
    prefixes = [512, 0, 0, 0, 0, 0, 0, 0]
    kv_lens_np = np.array([p + q for p, q in zip(prefixes, q_lens)], dtype=np.int32)
    cu_q_np = np.concatenate([[0], np.cumsum(q_lens, dtype=np.int32)])
    cu_kv_np = np.arange(bs + 1, dtype=np.int32) * (pps * page_size)
    page_np = np.arange(bs * pps, dtype=np.int32)

    chunks_t = ((0, 4096), (4096, 10240), (10240, 16384))
    expected_q = [
        [4096],  # Chunk 0: only seq 0
        [3404, 2740],  # Chunk 1: tail of seq 0 + head of seq 1
        [4760, 1384],  # Chunk 2: tail of seq 1 + pad seq 2
    ]
    expected_kv = [
        [512 + 4096],
        [512 + 7500, 2740],
        [7500, 1384],
    ]
    expected_pages_row0 = [0, 0, 1]

    class _Md:
        pass

    md = _Md()
    md.kv_lens = torch.from_numpy(kv_lens_np)
    md.cu_q_lens = torch.from_numpy(cu_q_np)
    md.cu_kv_lens = torch.from_numpy(cu_kv_np)
    md.page_indices = torch.from_numpy(page_np)
    md.swa_page_indices = torch.from_numpy(page_np + 10000)
    md.swa_group_page_indices = None

    metas_t = build_chunk_rpa_metas_torch(md, tp_size=8, chunks=(512, 768, 768))
    for k, (ts, te) in enumerate(chunks_t):
        kv_j, cuq_j, cukv_j, pg_j, dist_j = slice_chunk_rpa_metadata_jax(
            jnp.asarray(kv_lens_np),
            jnp.asarray(cu_q_np),
            jnp.asarray(cu_kv_np),
            jnp.asarray(page_np),
            t_start=ts,
            t_end=te,
        )
        n_act = int(dist_j[2])
        assert n_act == len(expected_q[k])
        q_act = np.diff(np.asarray(cuq_j)[: n_act + 1]).tolist()
        assert q_act == expected_q[k]
        assert np.asarray(kv_j)[:n_act].tolist() == expected_kv[k]
        assert int(pg_j[0]) == expected_pages_row0[k] * pps

        # Cross-check torch host metadata builder matches JAX slice identically.
        mt = metas_t[k]
        assert int(mt.distribution[2]) == n_act
        assert mt.kv_lens[:n_act].tolist() == expected_kv[k]
        assert np.diff(mt.cu_q_lens[: n_act + 1].numpy()).tolist() == expected_q[k]
        assert int(mt.swa_page_indices[0]) == expected_pages_row0[k] * pps + 10000


def test_causal_chunk_a1_rpa_interpret_parity_single_and_multiseq():
    """Verify 0-ULP / bit-exact causal parity of chunked A1 + RPAm + KV write vs unchunked reference."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from sglang_tpu.attention.pallas import ragged_paged_attention_v3 as v3
    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.kernel_d import (
        pallas_combine_unpool_residual_nextnorm_4slab,
    )
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        _d_with_next_in_gather,
    )

    # 1-rank interpret test with T=512 (chunks=(128, 192, 192), >=128 for A1 & >=128 for RPAm),
    # testing both single-seq Full and multi-seq packed SWA (seq0=320, seq1=192)!
    t_total = 512
    chunks = (128, 192, 192)
    h_dim = 256  # scaled H=256 for fast A1/D interpret
    h_q_local, h_kv_local = 16, 2
    q_size, kv_size = h_q_local * 64, h_kv_local * 64
    page_size = 128
    pps = t_total // page_size
    total_pages = pps + 4

    rng = np.random.default_rng(777)
    attn_in_np = (rng.standard_normal((t_total, h_dim)) * 0.4).astype(np.float32)
    wqkvg_np = (rng.standard_normal((h_dim, 2 * q_size + 2 * kv_size)) * 0.1).astype(
        np.float32
    )
    wqkvg_sc_np = np.full((1, 2 * q_size + 2 * kv_size), 0.02, dtype=np.float32)
    theta = rng.uniform(0.0, 2.0, size=(t_total, 128)).astype(np.float32)
    cos_np, sin_np = np.cos(theta), np.sin(theta)

    # Verify Kernel D row split with c0=128 matches full next_in on 1-rank interpret.
    keys = jax.random.split(jax.random.key(42), 7)
    d_args = (
        jax.random.normal(keys[0], (512, 32, 128), jnp.float32).astype(jnp.bfloat16),
        jax.random.randint(keys[1], (t_total, 8), 0, 512, jnp.int32),
        jax.random.uniform(keys[2], (t_total, 8), jnp.float32).astype(jnp.bfloat16),
        jax.random.normal(keys[3], (t_total, 8192), jnp.float32),
        (jax.random.normal(keys[4], (4096, 8192), jnp.float32) * 0.1).astype(
            jnp.float8_e4m3fn
        ),
        jnp.full((8192,), 0.02, jnp.float32),
        jax.random.normal(keys[5], (8192,), jnp.float32) * 0.1,
        jnp.full((8192,), 0.9, jnp.float32),
        jnp.full((8192,), 1.1, jnp.float32),
        jax.random.normal(keys[6], (8192,), jnp.float32) * 0.1,
    )
    d_fn = functools.partial(pallas_combine_unpool_residual_nextnorm_4slab, tok_tile=64)
    mesh = jax.sharding.Mesh(np.array(jax.devices("cpu")[:1]), ("model",))
    P = jax.sharding.PartitionSpec

    def _d_full(*xs):
        return d_fn(*(x[0] for x in xs))[1][None]

    def _d_split(*xs):
        _, rest, ag0 = _d_with_next_in_gather(
            d_fn=d_fn,
            d_args=tuple(x[0] for x in xs),
            rows0=chunks[0],
            split=True,
            axis_name="model",
            gather_static_fp8=True,
        )
        return rest[None], ag0[None]

    with _interpret_on_cpu():
        stacked_d = tuple(a[None] for a in d_args)
        ref_d_next = jax.jit(
            jax.shard_map(
                _d_full,
                mesh=mesh,
                in_specs=(P("model"),) * 10,
                out_specs=P("model"),
                check_vma=False,
            )
        )(*stacked_d)[0]
        rest_d, ag0_d = jax.jit(
            jax.shard_map(
                _d_split,
                mesh=mesh,
                in_specs=(P("model"),) * 10,
                out_specs=(P("model"), P("model")),
                check_vma=False,
            )
        )(*stacked_d)
        ref_d_fp8 = jnp.clip(
            ref_d_next.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
            -448.0,
            448.0,
        ).astype(jnp.float8_e4m3fn)
        got_d_fp8 = jnp.concatenate(
            [
                ag0_d[0, 0],
                jnp.clip(
                    rest_d[0].astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
                    -448.0,
                    448.0,
                ).astype(jnp.float8_e4m3fn),
            ],
            axis=0,
        )
        assert np.array_equal(
            np.asarray(ref_d_fp8).view(np.uint8), np.asarray(got_d_fp8).view(np.uint8)
        )

    # Now test A1 (rpa_layout=False, no VMEM bitcast in interpret) + causal chunked attention + KV write
    # across (Full 1-seq) and (SWA 2-seq packed)!
    cases = [
        ("full_1seq", None, [512], [512]),
        ("swa_2seq_split", 129, [320, 192], [320, 192]),
    ]
    for cname, sw, q_lens_l, kv_lens_l in cases:
        bs = len(q_lens_l)
        kv_lens_arr = jnp.asarray(kv_lens_l, dtype=jnp.int32)
        cu_q_arr = jnp.asarray([0] + list(np.cumsum(q_lens_l)), dtype=jnp.int32)
        cu_kv_arr = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size)
        page_arr = jnp.arange(bs * pps, dtype=jnp.int32)
        kvc_init = jnp.zeros((total_pages, page_size, h_kv_local, 64), jnp.bfloat16)

        x_j = jnp.asarray(attn_in_np, jnp.bfloat16)
        w_j = jnp.asarray(wqkvg_np, jnp.float8_e4m3fn)
        ws_j = jnp.asarray(wqkvg_sc_np, jnp.float32)
        cos_j = jnp.asarray(cos_np, jnp.float32)
        sin_j = jnp.asarray(sin_np, jnp.float32)

        def _rpa_ref_and_write(q, kv, k_pages, v_pages, kv_lens_i, cu_q_i, pg_i, n_act):
            q_3d = q.reshape(-1, h_q_local, 64)
            k_new = kv[:, :kv_size].astype(jnp.bfloat16).reshape(-1, h_kv_local, 64)
            v_new = kv[:, kv_size:].astype(jnp.bfloat16).reshape(-1, h_kv_local, 64)
            # Write new K/V tokens into paged k_pages, v_pages at [kv_len - q_len : kv_len].
            kp_flat = k_pages.reshape(-1, h_kv_local, 64)
            vp_flat = v_pages.reshape(-1, h_kv_local, 64)
            for s in range(n_act):
                q0 = int(cu_q_i[s])
                q1 = int(cu_q_i[s + 1])
                ql = q1 - q0
                kvl = int(kv_lens_i[s])
                kv0 = kvl - ql
                pg_base = s * pps
                for j in range(ql):
                    pos = kv0 + j
                    slot = int(pg_i[pg_base + pos // page_size]) * page_size + (pos % page_size)
                    kp_flat = kp_flat.at[slot].set(k_new[q0 + j])
                    vp_flat = vp_flat.at[slot].set(v_new[q0 + j])
            k_pages_u = kp_flat.reshape(total_pages, page_size, h_kv_local, 64)
            v_pages_u = vp_flat.reshape(total_pages, page_size, h_kv_local, 64)
            pg_2d = pg_i.reshape(bs, pps)
            out = v3.ref_ragged_paged_attention(
                q_3d,
                k_pages_u,
                v_pages_u,
                kv_lens_i,
                pg_2d,
                cu_q_i,
                jnp.array([n_act], dtype=jnp.int32),
                causal=True,
                sm_scale=1.0 / 8.0,
                sliding_window=sw,
            )
            return out.reshape(-1, q_size), k_pages_u, v_pages_u

        with _interpret_on_cpu():
            x_f8 = jnp.clip(
                x_j.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT, -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
            q_ref, g_ref, kv_ref = kernel_a1.pallas_a1_chunk_1way(
                x_f8[None],
                w_j,
                ws_j * kernel_a1.STATIC_A1_ACT_SCALE,
                cos_j[None],
                sin_j[None],
                None,
                t_shard=t_total,
                row_off=0,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=False,
                mode="all",
                tp_axis="model",
                tm=256,
            )
            ref_out, ref_kp, ref_vp = _rpa_ref_and_write(
                q_ref[0], kv_ref[0], kvc_init, kvc_init, kv_lens_arr, cu_q_arr, page_arr, bs
            )
            ref_gate = g_ref[0]

            # Chunked causal A1 + paged KV write + attention
            got_kp, got_vp = kvc_init, kvc_init
            out_parts = []
            gate_parts = []
            t0 = 0
            for c_k in chunks:
                t1 = t0 + c_k
                q_k, g_k, kv_k = kernel_a1.pallas_a1_chunk_1way(
                    x_f8[t0:t1][None],
                    w_j,
                    ws_j * kernel_a1.STATIC_A1_ACT_SCALE,
                    cos_j[t0:t1][None],
                    sin_j[t0:t1][None],
                    None,
                    t_shard=c_k,
                    row_off=0,
                    h_q_local=h_q_local,
                    h_kv_local=h_kv_local,
                    rpa_layout=False,
                    mode="all",
                    tp_axis="model",
                    tm=128 if c_k % 256 != 0 else 256,
                )
                kvl_k, cuq_k, _, pg_k, dst_k = slice_chunk_rpa_metadata_jax(
                    kv_lens_arr, cu_q_arr, cu_kv_arr, page_arr, t_start=t0, t_end=t1
                )
                out_k, got_kp, got_vp = _rpa_ref_and_write(
                    q_k[0], kv_k[0], got_kp, got_vp, kvl_k, cuq_k, pg_k, int(dst_k[2])
                )
                out_parts.append(out_k)
                gate_parts.append(g_k[0])
                t0 = t1
            got_out = jnp.concatenate(out_parts, axis=0)
            got_gate = jnp.concatenate(gate_parts, axis=0)

        assert np.array_equal(
            np.asarray(got_gate).view(np.uint16), np.asarray(ref_gate).view(np.uint16)
        ), f"{cname}: gate mismatch"
        assert np.array_equal(
            np.asarray(got_kp).view(np.uint16), np.asarray(ref_kp).view(np.uint16)
        ), f"{cname}: k_pages mismatch"
        assert np.array_equal(
            np.asarray(got_vp).view(np.uint16), np.asarray(ref_vp).view(np.uint16)
        ), f"{cname}: v_pages mismatch"
        rf = np.asarray(ref_out, dtype=np.float32)
        gf = np.asarray(got_out, dtype=np.float32)
        cos = float(
            np.dot(rf.ravel(), gf.ravel())
            / (np.linalg.norm(rf) * np.linalg.norm(gf) + 1e-12)
        )
        assert cos >= 0.999999, f"{cname}: cos={cos}"


def test_causal_chunk_stream_mosaic_tpu7x_export():
    """Verify TPU7x Mosaic StableHLO export for ag_chunk, a1_chunk, and 3-chunk moe_agd."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )

    tp = 8
    c0, c1, c2 = 512, 768, 768
    t_loc = c0 + c1 + c2  # 2048 -> T=16384
    h_q_local, h_kv_local = 16, 2
    n_qkvg = 2 * h_q_local * 64 + 2 * h_kv_local * 64
    devs = jax.devices("cpu")
    P = jax.sharding.PartitionSpec

    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
    ):
        mesh1 = jax.sharding.Mesh(np.array([devs[0]]), ("model",))

        def _ag_a1_chunk(x_c1, wqkvg, w_sc, cos_c1, sin_c1):
            x_f8 = jnp.clip(
                x_c1.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT, -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
            ag = jax.lax.all_gather(x_f8, "model", axis=0, tiled=False)
            q_k, gate_k, kv_k = kernel_a1.pallas_a1_chunk_1way(
                ag,
                wqkvg,
                w_sc * kernel_a1.STATIC_A1_ACT_SCALE,
                cos_c1.reshape(1, c1, 128),
                sin_c1.reshape(1, c1, 128),
                None,
                t_shard=c1,
                row_off=0,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=True,
                mode="all",
                tp_axis="model",
                tm=256,
            )
            return q_k, gate_k, kv_k

        sm = jax.shard_map(
            _ag_a1_chunk,
            mesh=mesh1,
            in_specs=(P("model"), P("model"), P("model"), P("model"), P("model")),
            out_specs=(P("model"), P("model"), P("model")),
            check_vma=False,
        )
        lowered_a1 = (
            jax.jit(sm)
            .trace(
                jax.ShapeDtypeStruct((c1, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((8192, n_qkvg), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((1, n_qkvg), jnp.float32),
                jax.ShapeDtypeStruct((c1, 128), jnp.float32),
                jax.ShapeDtypeStruct((c1, 128), jnp.float32),
            )
            .lower(lowering_platforms=("tpu",))
        )
        mlir_text = lowered_a1.as_text("stablehlo")
        assert "partition_id" not in mlir_text
        assert "tpu.custom_call" in mlir_text or "stablehlo.custom_call" in mlir_text

        if len(devs) >= tp:
            import os
            from sglang_tpu.attention.pallas import ragged_paged_attention_v3 as v3
            from sglang_tpu.attention_pallas_rpa_v3 import _invoke_v3_kernel

            mesh8 = jax.sharding.Mesh(np.array(devs[:tp]), ("model",))

            # Full 8-device TPU7x export of causal_chunk_a1_rpa_step_jax (rpa_layout=True + m0_ring_fuse)
            def _ccs_attn_8dev(ag0, x_rest, wq, wqs, cs, sn, kvc, kvl, pg, cuq, cukv):
                t0 = 0
                cos_cs, sin_cs, metas = [], [], []
                for c_k in (c0, c1, c2):
                    t1 = t0 + tp * c_k
                    cos_cs.append(cs[t0:t1].reshape(tp, c_k, 128))
                    sin_cs.append(sn[t0:t1].reshape(tp, c_k, 128))
                    metas.append(
                        slice_chunk_rpa_metadata_jax(
                            kvl, cuq, cukv, pg, t_start=t0, t_end=t1
                        )
                    )
                    t0 = t1

                def _rpa_c(q5_k, kv4_k, kvc_in, mk):
                    kvl_k, cuq_k, cukv_k, pg_k, dst_k = mk
                    return _invoke_v3_kernel(
                        v3.ragged_paged_attention,
                        queries=q5_k,
                        keys=kv4_k,
                        values=None,
                        kv_cache_fused=kvc_in,
                        kv_lens=kvl_k,
                        page_indices=pg_k,
                        cu_q_lens=cuq_k,
                        cu_kv_lens=cukv_k,
                        distribution=dst_k,
                        attention_sink=None,
                        sm_scale=1.0 / 8.0,
                        sliding_window=2049,
                        soft_cap=None,
                        k_scale=1.0,
                        v_scale=1.0,
                        decode_query_size=1,
                        use_causal_mask=True,
                        prepared_inputs=True,
                        a3_out_layout=2,
                    )

                return causal_chunk_a1_rpa_step_jax(
                    ag0,
                    x_rest,
                    wq,
                    wqs,
                    tuple(cos_cs),
                    tuple(sin_cs),
                    kvc,
                    tuple(metas),
                    _rpa_c,
                    chunks=(c0, c1, c2),
                    h_q_local=h_q_local,
                    h_kv_local=h_kv_local,
                    tp_axis="model",
                    use_xor1_ag=False,
                )

            with mock.patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
                    "SGLANG_TPU_RPA_V3_PREFILL_A3_LAYOUT": "1",
                    "SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE": "1",
                },
            ):
                sm_ccs = jax.shard_map(
                    _ccs_attn_8dev,
                    mesh=mesh8,
                    in_specs=(
                        P("model"),
                        P("model"),
                        P("model"),
                        P("model"),
                        P(),
                        P(),
                        P("model"),
                        P(),
                        P(),
                        P(),
                        P(),
                    ),
                    out_specs=(P(), P(), P("model")),
                    check_vma=False,
                )
                lowered_ccs = (
                    jax.jit(sm_ccs)
                    .trace(
                        jax.ShapeDtypeStruct((tp * tp, c0, 8192), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((tp * (c1 + c2), 8192), jnp.bfloat16),
                        jax.ShapeDtypeStruct((tp * 8192, n_qkvg), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((tp, n_qkvg), jnp.float32),
                        jax.ShapeDtypeStruct((tp * t_loc, 128), jnp.float32),
                        jax.ShapeDtypeStruct((tp * t_loc, 128), jnp.float32),
                        jax.ShapeDtypeStruct((tp * 136, 128, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((8,), jnp.int32),
                        jax.ShapeDtypeStruct((8 * 128,), jnp.int32),
                        jax.ShapeDtypeStruct((9,), jnp.int32),
                        jax.ShapeDtypeStruct((9,), jnp.int32),
                    )
                    .lower(lowering_platforms=("tpu",))
                )
                ccs_mlir = lowered_ccs.as_text("stablehlo")
                assert "partition_id" not in ccs_mlir
                assert "pallas_a1" in ccs_mlir
                assert "RPAm" in ccs_mlir

            def _moe_agd_512(*xs):
                outs = fused_moe_full_bridge_and_ffn(
                    *xs,
                    mesh=mesh8,
                    router_mode="bf16_hilo",
                    b2_layout="nstream",
                    b2_two_half=True,
                    b2_deq0_first=True,
                    hoist_routing=True,
                    d_layout="4slab",
                    d_tok_tile=64,
                    in_shard_map=True,
                    b2_deq0_new_buffer=True,
                    next_in_gather_rows=c0,
                    next_in_gather_static_fp8=True,
                )
                return outs[0], outs[4], outs[1]

            sm_moe = jax.shard_map(
                _moe_agd_512,
                mesh=mesh8,
                in_specs=(
                    P("model"),
                    P("model"),
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
                out_specs=(P("model"), P("model"), P("model")),
                check_vma=False,
            )
            lowered_moe = (
                jax.jit(sm_moe)
                .trace(
                    jax.ShapeDtypeStruct((tp * t_loc, 8192), jnp.bfloat16),
                    jax.ShapeDtypeStruct((tp * t_loc, 8192), jnp.float32),
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
                    jax.ShapeDtypeStruct((tp,), jnp.int32),
                    jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.float32),
                    jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                )
                .lower(lowering_platforms=("tpu",))
            )
            assert "partition_id" not in lowered_moe.as_text("stablehlo")

        ids = chunk_major_global_token_ids_jax(tp, (c0, c1, c2))
        assert ids.shape == (tp, t_loc)
        x_sample = jnp.arange(tp * t_loc, dtype=jnp.int32)[:, None]
        cm = permute_to_chunk_major_jax(x_sample, tp, (c0, c1, c2))
        rec = unpermute_from_chunk_major_jax(cm, tp, (c0, c1, c2))
        assert np.array_equal(np.asarray(rec), np.asarray(x_sample))


def test_causal_chunk_packed_metadata_and_a3_xor_equivariance():
    """Verify packed 2x8k metadata slicing parity and A3 v16 XOR weight equivariance."""
    import jax
    import jax.numpy as jnp
    import torch

    from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import ChunkRpaMetaTorch

    tp = 8
    q_lens = [7500, 7500, 1384]
    pps = 64
    pg_flat = np.arange(1, len(q_lens) * pps + 1, dtype=np.int32)
    kvl_t = torch.tensor(q_lens, dtype=torch.int32)
    cuq_t = torch.tensor([0] + list(np.cumsum(q_lens)), dtype=torch.int32)
    cukv_t = torch.arange(len(q_lens) + 1, dtype=torch.int32) * (pps * 128)
    pg_t = torch.tensor(pg_flat, dtype=torch.int32)
    md = ChunkRpaMetaTorch(
        kv_lens=kvl_t,
        cu_q_lens=cuq_t,
        cu_kv_lens=cukv_t,
        page_indices=pg_t,
        swa_page_indices=None,
        swa_group_page_indices=None,
        distribution=torch.tensor([0, 0, len(q_lens)], dtype=torch.int32),
    )

    chunks = (512, 768, 768)
    torch_metas = build_chunk_rpa_metas_torch(md, tp_size=tp, chunks=chunks)
    kvl_j = jnp.asarray(q_lens, dtype=jnp.int32)
    cuq_j = jnp.asarray([0] + list(np.cumsum(q_lens)), dtype=jnp.int32)
    cukv_j = jnp.asarray(np.arange(len(q_lens) + 1, dtype=np.int32) * (pps * 128))
    pg_j = jnp.asarray(pg_flat, dtype=jnp.int32)
    with jax.default_device(jax.devices("cpu")[0]):
        t0 = 0
        for k, c_k in enumerate(chunks):
            t1 = t0 + tp * c_k
            kv_j, cq_j, ckv_j, p_j, d_j = slice_chunk_rpa_metadata_jax(
                kvl_j, cuq_j, cukv_j, pg_j, t_start=t0, t_end=t1
            )
            assert np.array_equal(np.asarray(kv_j), torch_metas[k].kv_lens.numpy())
            assert np.array_equal(np.asarray(cq_j), torch_metas[k].cu_q_lens.numpy())
            assert np.array_equal(np.asarray(ckv_j), torch_metas[k].cu_kv_lens.numpy())
            assert np.array_equal(np.asarray(p_j), torch_metas[k].page_indices.numpy())
            assert np.array_equal(np.asarray(d_j), torch_metas[k].distribution.numpy())
            t0 = t1

    # Regression for run_w7o_exos.py `wo_s` bug: A3 v16 step d on rank r consumes head group (r ^ d)
    # with weight slice d, so wo_xor[r, d] MUST equal wo_full[r ^ d] for rank-migration invariance.
    rng = np.random.default_rng(7)
    t_shard = 1024
    t_tot = tp * t_shard
    k_loc, h_out = 16, 32
    ao_full = rng.standard_normal((t_tot, tp, k_loc)).astype(np.float32)
    wo_full = rng.standard_normal((tp, k_loc, h_out)).astype(np.float32)

    def _run_a3_xor_sim(ao_perm, wo_per_rank):
        out = np.zeros((t_tot, h_out), dtype=np.float32)
        for r in range(tp):
            r0, r1 = r * t_shard, (r + 1) * t_shard
            acc = np.zeros((t_shard, h_out), dtype=np.float32)
            for d in range(tp):
                peer = r ^ d
                acc += ao_perm[r0:r1, peer, :] @ wo_per_rank[r, d, :, :]
            out[r0:r1] = acc
        return out

    wo_xor = np.stack([wo_full[[r ^ d for d in range(tp)]] for r in range(tp)], axis=0)
    wo_rep = np.broadcast_to(wo_full[None], (tp, tp, k_loc, h_out))

    ref = _run_a3_xor_sim(ao_full, wo_xor)
    ref_rep = _run_a3_xor_sim(ao_full, wo_rep)
    for c_test in ((512, 512), (256, 384, 384)):
        perm = np.asarray(to_chunk_major_perm(tp, c_test), dtype=np.int32)
        inv = np.asarray(from_chunk_major_perm(tp, c_test), dtype=np.int32)
        got_xor = _run_a3_xor_sim(ao_full[perm], wo_xor)[inv]
        assert np.allclose(got_xor, ref, atol=1e-5, rtol=1e-5)

        got_rep = _run_a3_xor_sim(ao_full[perm], wo_rep)[inv]
        same_rank_frac = float(np.mean((perm // t_shard) == (np.arange(t_tot) // t_shard)))
        cos_rep = float(
            np.dot(got_rep.ravel(), ref_rep.ravel())
            / (np.linalg.norm(got_rep) * np.linalg.norm(ref_rep) + 1e-12)
        )
        assert abs(cos_rep - same_rank_frac) < 0.05


def test_causal_chunk_skip_pad_moe_and_eplb_calib_masking():
    """Verify SKIP_PAD_MOE_TOKENS=1 + EPLB calib masking under chunk-major CAUSAL_CHUNK_STREAM."""
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from sglang_tpu.muse_prefill_kernels.eplb import build_eplb_moe_op
    from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels.pipeline import (
        fused_moe_full_bridge_and_ffn,
    )

    devs = jax.devices("cpu")
    if len(devs) < 8:
        pytest.skip("Requires >=8 JAX CPU devices")

    tp = 8
    t_loc = 1024
    t_tot = tp * t_loc
    chunks = resolve_causal_chunks(t_loc)  # (256, 384, 384)
    assert chunks == (256, 384, 384)
    # vt=4500 leaves real tokens [4096..4500) in Chunk 1 on ranks 5..6 and puts real tokens
    # from Chunk 0 on rank 7 at chunk-major indices >= 7*1024, while padding lands on rank 0 < 4500.
    vt = 4500
    perm = np.asarray(to_chunk_major_perm(tp, chunks), dtype=np.int32)
    inv = np.asarray(from_chunk_major_perm(tp, chunks), dtype=np.int32)

    tok_ids_2d = chunk_major_global_token_ids_jax(tp, chunks)
    assert np.array_equal(np.asarray(tok_ids_2d).reshape(-1), perm)
    # Confirm rank tp-1 has real tokens at linear chunk-major indices >= vt (the old bug trigger).
    assert np.any((perm[7 * t_loc :] < vt))

    rng = np.random.default_rng(123)
    raw_ids_np = rng.integers(0, 256, size=(t_tot, 8), dtype=np.int32)
    tw_np = rng.uniform(0.05, 0.5, size=(t_tot, 8)).astype(np.float32)

    mesh8 = jax.sharding.Mesh(np.array(devs[:tp]), ("model",))

    def _mask_body(local_ids, local_tw, vt_arr, *, is_ccs: bool):
        my_id = lax.axis_index("model")
        vt_s = vt_arr.reshape(-1)[0]
        g_raw = lax.all_gather(local_ids, "model", axis=0, tiled=False).reshape(-1, 8)
        if is_ccs:
            t_2d = chunk_major_global_token_ids_jax(tp, chunks)
            g_act = t_2d.reshape(-1, 1) < vt_s
            l_act = lax.dynamic_slice_in_dim(t_2d, my_id, 1, axis=0).reshape(t_loc, 1) < vt_s
        else:
            g_act = jnp.arange(t_tot, dtype=jnp.int32)[:, None] < vt_s
            l_act = (my_id * t_loc + jnp.arange(t_loc, dtype=jnp.int32))[:, None] < vt_s
        sentinel = jnp.int32(256)
        g_masked = jnp.where(g_act, g_raw, sentinel)
        l_masked = jnp.where(l_act, local_ids, sentinel)
        tw_masked = jnp.where(l_act, local_tw, jnp.float32(0.0))
        expert_iota = jnp.arange(256, dtype=jnp.int32)[None, :]
        hist_in = jnp.where(l_act, local_ids, jnp.int32(-1)).reshape(-1, 1)
        g_hist = lax.psum(jnp.sum((hist_in == expert_iota).astype(jnp.int32), axis=0), "model")
        return g_masked[None], l_masked, tw_masked, g_hist[None]

    def _run_mask(ids_in, tw_in, *, is_ccs: bool):
        fn = jax.jit(
            jax.shard_map(
                lambda li, lt, v: _mask_body(li, lt, v, is_ccs=is_ccs),
                mesh=mesh8,
                in_specs=(P("model"), P("model"), P()),
                out_specs=(P("model"), P("model"), P("model"), P("model")),
                check_vma=False,
            )
        )
        return fn(
            jnp.asarray(ids_in),
            jnp.asarray(tw_in),
            jnp.asarray([vt], dtype=jnp.int32),
        )

    g_ctrl, l_ctrl, tw_ctrl, h_ctrl = _run_mask(raw_ids_np, tw_np, is_ccs=False)
    g_ccs, l_ccs, tw_ccs, h_ccs = _run_mask(raw_ids_np[perm], tw_np[perm], is_ccs=True)

    assert np.array_equal(np.asarray(g_ccs[0])[inv], np.asarray(g_ctrl[0]))
    assert np.array_equal(np.asarray(l_ccs)[inv], np.asarray(l_ctrl))
    assert np.allclose(np.asarray(tw_ccs)[inv], np.asarray(tw_ctrl), atol=1e-6)
    assert np.array_equal(np.asarray(h_ccs[0]), np.asarray(h_ctrl[0]))
    assert int(np.asarray(h_ccs[0]).sum()) == vt * 8

    # Verify TPU7x StableHLO lowering of fused_moe_full_bridge_and_ffn(valid_tokens=...,
    # causal_chunk_stream=True, next_in_gather_rows=c0) and build_eplb_moe_op wiring.
    with (
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        ),
        tpu_info_for_cpu_export(),
    ):
        with pytest.raises(ValueError, match="online prefill EPLB calibration"):
            build_eplb_moe_op(
                h_q_local=16,
                h_kv_local=2,
                fused_glue=True,
                eplb_mode="palette_ep",
                eplb_calib=True,
                skip_pad_tokens=True,
                next_in_gather_rows_fn=lambda ts: resolve_causal_chunks(ts)[0],
                causal_chunk_stream=True,
            )

        captured_eplb = {}

        def _fake_export(op_name, fn, **kwargs):
            captured_eplb["op_name"] = op_name
            captured_eplb["fn"] = fn
            return fn

        with (
            mock.patch(
                "sglang_tpu.moe.pallas.fused_moe_rs_op._get_mesh", return_value=mesh8
            ),
            mock.patch(
                "sglang_tpu.pallas_jax_op.cpu_export_jax_op", side_effect=_fake_export
            ),
        ):
            build_eplb_moe_op(
                h_q_local=16,
                h_kv_local=2,
                fused_glue=True,
                eplb_mode="",
                eplb_calib=False,
                skip_pad_tokens=True,
                next_in_gather_rows_fn=lambda ts: resolve_causal_chunks(ts)[0],
                causal_chunk_stream=True,
            )
        assert "_ccs" in captured_eplb["op_name"]
        assert "_spad" in captured_eplb["op_name"]

        sm_moe_spad = jax.shard_map(
            lambda *xs: fused_moe_full_bridge_and_ffn(
                *xs[:22],
                mesh=mesh8,
                router_mode="bf16_hilo",
                b2_layout="nstream",
                b2_two_half=True,
                b2_deq0_first=True,
                hoist_routing=True,
                d_layout="4slab",
                d_tok_tile=64,
                in_shard_map=True,
                b2_deq0_new_buffer=True,
                valid_tokens=xs[22],
                next_in_gather_rows=chunks[0],
                next_in_gather_static_fp8=True,
                causal_chunk_stream=True,
            )[0],
            mesh=mesh8,
            in_specs=(
                P("model"),
                P("model"),
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
                P(),
            ),
            out_specs=P("model"),
            check_vma=False,
        )
        hlo_spad = (
            jax.jit(sm_moe_spad)
            .trace(
                jax.ShapeDtypeStruct((t_tot, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((t_tot, 8192), jnp.float32),
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
                jax.ShapeDtypeStruct((tp,), jnp.int32),
                jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
                jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
                jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            )
            .lower(lowering_platforms=("tpu",))
            .as_text("stablehlo")
        )
        assert "partition_id" not in hlo_spad
