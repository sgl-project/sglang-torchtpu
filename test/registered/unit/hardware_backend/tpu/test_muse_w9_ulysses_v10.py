# SPDX-License-Identifier: Apache-2.0
"""CPU 16-device interpret=True (uninitialized_memory='nan') parity, L61-prune,
and TPU7x Mosaic export tests for Static-FP8 Ulysses Attention (`A1_VARIANT='v10_sp'`).
"""

from __future__ import annotations

import ast
import contextlib
import pathlib
from unittest import mock

import numpy as np
import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.pallas_export import tpu_info_for_cpu_export

_TP = 16
_AXIS = "model"


@contextlib.contextmanager
def _interpret_on_cpu(mesh=None):
    import jax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    real = pl.pallas_call

    def interpret(*args, **kwargs):
        kwargs["interpret"] = pltpu.InterpretParams(uninitialized_memory="nan")
        return real(*args, **kwargs)

    ctx_stack = contextlib.ExitStack()
    ctx_stack.enter_context(
        mock.patch(
            "sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"
        )
    )
    ctx_stack.enter_context(tpu_info_for_cpu_export())
    ctx_stack.enter_context(
        mock.patch("sglang_tpu.attention.pallas.util.get_tpu_version", return_value=7)
    )
    ctx_stack.enter_context(
        mock.patch(
            "sglang_tpu.attention.pallas.ragged_paged_attention_v3.get_tpu_version",
            return_value=7,
        )
    )
    ctx_stack.enter_context(mock.patch.object(pl, "pallas_call", interpret))
    if mesh is not None:
        ctx_stack.enter_context(
            mock.patch(
                "sglang_tpu.moe.pallas.fused_moe_rs_op._get_mesh", return_value=mesh
            )
        )
    ctx_stack.enter_context(jax.default_device(jax.devices("cpu")[0]))
    with ctx_stack:
        yield


def _fp8_k_major_np(w_nk: np.ndarray):
    w = w_nk.T.astype(np.float32)
    scale = np.maximum(np.max(np.abs(w), axis=0) / 448.0, 1e-12).astype(np.float32)
    return (w / scale[None, :]), scale


def test_v10_sp_16dev_interpret_nan_parity_vs_v6():
    """16-device CPU interpret parity (`uninitialized_memory='nan'`) of v10_sp vs v6
    on `1536` local rows (`512 + 1024` chunks -> `T = 24,576`) covering Q, K, V,
    gate, normalized attention out, and final Wo out.
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax import lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    if len(jax.devices("cpu")) < _TP:
        pytest.skip(f"Need {_TP} CPU devices (got {len(jax.devices('cpu'))})")

    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1, kernel_a3
    from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import to_chunk_major_perm

    devs = jax.devices("cpu")[:_TP]
    mesh = Mesh(np.array(devs), (_AXIS,))

    chunks = (512, 1024)
    t_shard = sum(chunks)  # 1536
    t_total = _TP * t_shard  # 24576
    h_dim = 256  # Scaled H=256 so 16-dev Python interpret finishes quickly while T=24576 & Q=512 match production
    h_q_local = 8
    h_kv_local = 1
    q_size = h_q_local * 64  # 512
    kv_size = h_kv_local * 64  # 64
    n_full = _TP * q_size  # 8192

    rng = np.random.default_rng(2026)
    attn_in_np = (rng.standard_normal((t_total, h_dim)) * 0.35).astype(np.float32)
    q_w_np = (rng.standard_normal((_TP, q_size, h_dim)) * 0.2).astype(np.float32)
    k_w_np = (rng.standard_normal((_TP, kv_size, h_dim)) * 0.2).astype(np.float32)
    v_w_np = (rng.standard_normal((_TP, kv_size, h_dim)) * 0.2).astype(np.float32)
    g_w_np = (rng.standard_normal((_TP, q_size, h_dim)) * 0.2).astype(np.float32)
    wo_w_np = (rng.standard_normal((h_dim, _TP, q_size)) * 0.15).astype(np.float32)

    theta = rng.uniform(0.0, 2.0, size=(t_total, 128)).astype(np.float32)
    cos_np = np.cos(theta)
    sin_np = np.sin(theta)

    perm_cm = np.array(to_chunk_major_perm(_TP, chunks), dtype=np.int64)
    attn_in_shard_np = attn_in_np[perm_cm].reshape(_TP, t_shard, h_dim)

    v6_wqkvg_list, v6_wqkvg_sc_list = [], []
    ul_wqkv_list, ul_wqkv_sc_list = [], []
    w_gate_parts, w_gate_sc_parts = [], []
    for r in range(_TP):
        q_r, k_r, v_r, g_r = q_w_np[r], k_w_np[r], v_w_np[r], g_w_np[r]
        w_v6, sc_v6 = _fp8_k_major_np(np.concatenate([q_r, g_r, k_r, v_r], axis=0))
        v6_wqkvg_list.append(w_v6)
        v6_wqkvg_sc_list.append(sc_v6[None, :])
        w_qkv, sc_qkv = _fp8_k_major_np(np.concatenate([q_r, k_r, v_r], axis=0))
        ul_wqkv_list.append(w_qkv)
        ul_wqkv_sc_list.append(sc_qkv)
        w_g, sc_g = _fp8_k_major_np(g_r)
        w_gate_parts.append(w_g)
        w_gate_sc_parts.append(sc_g)

    w_qkv_3d_np = np.stack(ul_wqkv_list, axis=0)
    w_qkv_sc_3d_np = np.stack(ul_wqkv_sc_list, axis=0)[:, None, :]
    w_gate_full_np = np.concatenate(w_gate_parts, axis=1)
    w_gate_sc_np = np.concatenate(w_gate_sc_parts, axis=0)[None, :]
    wo_full_np, wo_sc_full_np = _fp8_k_major_np(wo_w_np.reshape(h_dim, n_full))

    def _put(arr, spec, dt):
        return jax.device_put(jnp.asarray(arr, dtype=dt), NamedSharding(mesh, spec))

    with _interpret_on_cpu(mesh):
        attn_in_d = _put(attn_in_shard_np, P(_AXIS), jnp.bfloat16)
        v6_w_d = _put(np.stack(v6_wqkvg_list, 0), P(_AXIS), jnp.float8_e4m3fn)
        v6_sc_d = _put(np.stack(v6_wqkvg_sc_list, 0), P(_AXIS), jnp.float32)

        ul_w_d = _put(
            np.broadcast_to(w_qkv_3d_np[None], (_TP, *w_qkv_3d_np.shape)),
            P(_AXIS),
            jnp.float8_e4m3fn,
        )
        ul_sc_d = _put(
            np.broadcast_to(w_qkv_sc_3d_np[None], (_TP, *w_qkv_sc_3d_np.shape)),
            P(_AXIS),
            jnp.float32,
        )
        w_gate_d = _put(
            np.broadcast_to(w_gate_full_np[None], (_TP, *w_gate_full_np.shape)),
            P(_AXIS),
            jnp.float8_e4m3fn,
        )
        w_gate_sc_d = _put(
            np.broadcast_to(w_gate_sc_np[None], (_TP, *w_gate_sc_np.shape)),
            P(_AXIS),
            jnp.float32,
        )

        wo_v16_list = []
        wo_raw_3d = wo_full_np.reshape(_TP, q_size, h_dim)
        for r in range(_TP):
            order = [
                kernel_a3.a3_v16_step_peer(r, d, _TP, parity_stagger=False)
                for d in range(_TP)
            ]
            wo_v16_list.append(wo_raw_3d[order])
        wo_v16_d = _put(np.stack(wo_v16_list, 0), P(_AXIS), jnp.float8_e4m3fn)
        wo_seq_d = _put(
            np.broadcast_to(wo_full_np[None], (_TP, *wo_full_np.shape)),
            P(_AXIS),
            jnp.float8_e4m3fn,
        )
        wo_sc_d = _put(
            np.broadcast_to(wo_sc_full_np[None, None, :], (_TP, 1, h_dim)),
            P(_AXIS),
            jnp.float32,
        )

        c_offs = (0, _TP * chunks[0])
        cos_chunks_d = [
            _put(
                np.broadcast_to(
                    cos_np[None, c_offs[k] : c_offs[k] + _TP * c_k],
                    (_TP, _TP * c_k, 128),
                ),
                P(_AXIS),
                jnp.float32,
            )
            for k, c_k in enumerate(chunks)
        ]
        sin_chunks_d = [
            _put(
                np.broadcast_to(
                    sin_np[None, c_offs[k] : c_offs[k] + _TP * c_k],
                    (_TP, _TP * c_k, 128),
                ),
                P(_AXIS),
                jnp.float32,
            )
            for k, c_k in enumerate(chunks)
        ]

        def _v6_chunk_fn(x_k, w6, sc6, cc, ss, *, c_k: int):
            w6 = w6[0]
            sc6 = sc6[0] * kernel_a1.STATIC_A1_ACT_SCALE
            x_fp8 = jnp.clip(
                x_k[0].astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT, -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
            ag_k = lax.all_gather(x_fp8, _AXIS, axis=0, tiled=False)
            q_k, g_k, kv_k = kernel_a1.pallas_a1_chunk_1way(
                ag_k,
                w6,
                sc6,
                cc[0].reshape(_TP, c_k, 128),
                ss[0].reshape(_TP, c_k, 128),
                None,
                t_shard=c_k,
                row_off=0,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=False,
                mode="all",
                tp_axis=_AXIS,
                tm=256,
            )
            c_tok = _TP * c_k
            q_2d = q_k.reshape(c_tok, q_size)
            g_2d = g_k.reshape(c_tok, q_size)
            kv_2d = kv_k.reshape(c_tok, 2 * kv_size)
            v_rec = kv_2d[:, kv_size:].astype(jnp.float32)
            attn_raw = (
                q_2d.astype(jnp.float32) * 0.2 + jnp.tile(v_rec, (1, h_q_local)) * 0.7
            ).astype(jnp.bfloat16)
            qp_k = kernel_a3.pallas_gate_static_fp8_quant(
                attn_raw, g_2d, tm=512, head_norm_eps=1e-5, rpa_packed_out=False
            ).reshape(_TP, c_k, q_size)
            q_a2a = lax.all_to_all(
                qp_k, _AXIS, split_axis=0, concat_axis=0, tiled=False
            ).reshape(_TP, c_k, q_size)
            return q_2d[None], kv_2d[None], g_2d[None], attn_raw[None], q_a2a[None]

        def _v10_qkv_fn(x_k, wul, scul, cc, ss, *, c_k: int):
            me = lax.axis_index(_AXIS)
            x_fp8 = jnp.clip(
                x_k[0].astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT, -448.0, 448.0
            ).astype(jnp.float8_e4m3fn)
            cos_own = lax.dynamic_index_in_dim(
                cc[0].reshape(_TP, c_k, 128), me, axis=0, keepdims=False
            )
            sin_own = lax.dynamic_index_in_dim(
                ss[0].reshape(_TP, c_k, 128), me, axis=0, keepdims=False
            )
            q_3d, kv_3d = kernel_a1.pallas_a1_ulysses_qkv_local(
                x_fp8,
                wul[0],
                scul[0] * kernel_a1.STATIC_A1_ACT_SCALE,
                cos_own,
                sin_own,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
            )
            c_tok = _TP * c_k
            q_2d = lax.all_to_all(
                q_3d, _AXIS, split_axis=0, concat_axis=0, tiled=False
            ).reshape(c_tok, q_size)
            kv_2d = lax.all_to_all(
                kv_3d, _AXIS, split_axis=0, concat_axis=0, tiled=False
            ).reshape(c_tok, kv_3d.shape[-1])
            return q_2d[None], kv_2d[None], x_fp8[None]

        def _v10_gate_fn(x_fp8, wg, wgc):
            return kernel_a1.pallas_ulysses_local_gate(
                x_fp8[0], wg[0], wgc[0] * kernel_a1.STATIC_A1_ACT_SCALE
            )[None]

        def _v10_rev_fuse_fn(attn_raw, g_loc, *, c_k: int):
            q_f8_3d = kernel_a3.pallas_ulysses_norm_fp8_for_rev_a2a(
                attn_raw[0], head_norm_eps=1e-5
            ).reshape(_TP, c_k, q_size)
            attn_fp8_3d = lax.all_to_all(
                q_f8_3d, _AXIS, split_axis=0, concat_axis=0, tiled=False
            ).reshape(_TP, c_k, q_size)
            return kernel_a3.pallas_ulysses_gate_fuse(attn_fp8_3d, g_loc[0])[None]

        def _v6_wo_ref_fn(qa0, qa1, woseq, wosc):
            q_a2a_3d = jnp.concatenate([qa0[0], qa1[0]], axis=1)
            return (
                jnp.einsum(
                    "rth,rhn->tn",
                    q_a2a_3d.astype(jnp.float32),
                    woseq[0].reshape(_TP, q_size, h_dim).astype(jnp.float32),
                )
                * (wosc[0] * kernel_a3.STATIC_A3_ACT_SCALE)
            ).astype(jnp.bfloat16)[None]

        def _v10_wo_fn(qp0, qp1, woseq, wosc):
            return kernel_a3.pallas_ulysses_wo_local(
                (qp0[0], qp1[0]), woseq[0], wosc[0], bn=min(1024, h_dim)
            )[None]

        ref_chunks, got_chunks, qa_v6, qp_v10 = [], [], [], []
        r_off = 0
        for k_idx, c_k in enumerate(chunks):
            x_slice_d = attn_in_d[:, r_off : r_off + c_k, :]
            r_off += c_k

            sm_v6_c = jax.shard_map(
                lambda xk, w6, sc6, cc, ss, _c=c_k: _v6_chunk_fn(xk, w6, sc6, cc, ss, c_k=_c),
                mesh=mesh,
                in_specs=(P(_AXIS),) * 5,
                out_specs=(P(_AXIS),) * 5,
                check_vma=False,
            )
            q_r, kv_r, g_r, attn_r, qa_r = jax.block_until_ready(
                sm_v6_c(x_slice_d, v6_w_d, v6_sc_d, cos_chunks_d[k_idx], sin_chunks_d[k_idx])
            )
            ref_chunks.append((q_r, kv_r, g_r))
            qa_v6.append(qa_r)

            sm_v10_qkv = jax.shard_map(
                lambda xk, wul, scul, cc, ss, _c=c_k: _v10_qkv_fn(xk, wul, scul, cc, ss, c_k=_c),
                mesh=mesh,
                in_specs=(P(_AXIS),) * 5,
                out_specs=(P(_AXIS),) * 3,
                check_vma=False,
            )
            q_g, kv_g, xfp8_g = jax.block_until_ready(
                sm_v10_qkv(x_slice_d, ul_w_d, ul_sc_d, cos_chunks_d[k_idx], sin_chunks_d[k_idx])
            )

            sm_v10_g = jax.shard_map(
                _v10_gate_fn,
                mesh=mesh,
                in_specs=(P(_AXIS),) * 3,
                out_specs=P(_AXIS),
                check_vma=False,
            )
            gloc_g = jax.block_until_ready(sm_v10_g(xfp8_g, w_gate_d, w_gate_sc_d))
            got_chunks.append((q_g, kv_g, gloc_g))

            sm_v10_rf = jax.shard_map(
                lambda ar, gl, _c=c_k: _v10_rev_fuse_fn(ar, gl, c_k=_c),
                mesh=mesh,
                in_specs=(P(_AXIS), P(_AXIS)),
                out_specs=P(_AXIS),
                check_vma=False,
            )
            qp_g = jax.block_until_ready(sm_v10_rf(attn_r, gloc_g))
            qp_v10.append(qp_g)

        sm_v6_wo = jax.shard_map(
            _v6_wo_ref_fn, mesh=mesh, in_specs=(P(_AXIS),) * 4, out_specs=P(_AXIS), check_vma=False
        )
        sm_v10_wo = jax.shard_map(
            _v10_wo_fn, mesh=mesh, in_specs=(P(_AXIS),) * 4, out_specs=P(_AXIS), check_vma=False
        )
        wo_ref_d = jax.block_until_ready(sm_v6_wo(qa_v6[0], qa_v6[1], wo_seq_d, wo_sc_d))
        wo_got_d = jax.block_until_ready(sm_v10_wo(qp_v10[0], qp_v10[1], wo_seq_d, wo_sc_d))
        ref = (
            ref_chunks[0][0], ref_chunks[1][0],
            ref_chunks[0][1], ref_chunks[1][1],
            ref_chunks[0][2], ref_chunks[1][2],
            wo_ref_d,
        )
        got = (
            got_chunks[0][0], got_chunks[1][0],
            got_chunks[0][1], got_chunks[1][1],
            got_chunks[0][2], got_chunks[1][2],
            wo_got_d,
        )

    for r in range(_TP):
        for k_idx in (0, 1):
            q_ref = np.asarray(ref[k_idx].addressable_shards[r].data[0], dtype=np.float32)
            q_got = np.asarray(got[k_idx].addressable_shards[r].data[0], dtype=np.float32)
            assert np.all(np.isfinite(q_got)), f"NaN/Inf in q chunk {k_idx} rank {r}"
            np.testing.assert_array_equal(q_got, q_ref)

            kv_ref_u8 = np.asarray(
                jax.lax.bitcast_convert_type(
                    ref[2 + k_idx].addressable_shards[r].data[0], jnp.uint8
                )
            )
            kv_got_u8 = np.asarray(
                jax.lax.bitcast_convert_type(
                    got[2 + k_idx].addressable_shards[r].data[0], jnp.uint8
                )
            )
            assert np.all(
                np.isfinite(
                    np.asarray(got[2 + k_idx].addressable_shards[r].data[0], dtype=np.float32)
                )
            )
            np.testing.assert_array_equal(kv_got_u8, kv_ref_u8)

            c_k = chunks[k_idx]
            g_got_r = np.asarray(
                got[4 + k_idx].addressable_shards[r].data[0], dtype=np.float32
            )
            assert np.all(np.isfinite(g_got_r))
            for h in range(_TP):
                g_ref_rh = np.asarray(
                    ref[4 + k_idx].addressable_shards[h].data[0][
                        r * c_k : (r + 1) * c_k
                    ],
                    dtype=np.float32,
                )
                np.testing.assert_array_equal(
                    g_got_r[:, h * q_size : (h + 1) * q_size], g_ref_rh
                )

    wo_ref = np.asarray(ref[6], dtype=np.float32).reshape(-1, h_dim)
    wo_got = np.asarray(got[6], dtype=np.float32).reshape(-1, h_dim)
    assert np.all(np.isfinite(wo_got))
    cos = float(
        np.sum(wo_ref * wo_got)
        / (np.linalg.norm(wo_ref) * np.linalg.norm(wo_got) + 1e-12)
    )
    rel_l2 = float(np.linalg.norm(wo_ref - wo_got) / (np.linalg.norm(wo_ref) + 1e-12))
    # 12 MB FP8 rev-A2A applies 2x static FP8 quantizations vs v6's 1x FP8 quant (~0.9989-0.9990 cos).
    assert cos >= 0.9985, f"Wo parity cos={cos:.6f} too low"
    assert rel_l2 <= 0.055, f"Wo parity rel_l2={rel_l2:.6f} too high"


def _assert_tpu7x_spmd_export_and_compile(op_def, fake_torch_args):
    """Verify zero `stablehlo.partition_id` in exported MLIR and full TPU7x SPMD XLA `.compile()`."""
    import jax
    from sglang_tpu.pallas_jax_op import _jax_placeholders

    fn_obj = op_def._init_fn
    jax_args = _jax_placeholders(
        fake_torch_args, mesh=fn_obj.mesh, partition_specs=fn_obj.input_partition_specs
    )
    with tpu_info_for_cpu_export(), jax._src.config.export_ignore_forward_compatibility(True):
        lowered = fn_obj.exported(*jax_args)
        assert lowered is not None
        assert "partition_id" not in lowered.mlir_module(), (
            f"stablehlo.partition_id leaked in {fn_obj.name}"
        )
        compiled = fn_obj.jit_fn.trace(*jax_args).lower().compile()
        assert compiled is not None


def test_v10_sp_tpu7x_2x2x2_full_prod_aot_compile():
    """Offline TPU v7 (`2x2x2` TP=16) Mosaic AOT compile + zero-`partition_id` check of ALL
    `v10_sp` ops at full Muse Spark production geometry (`H=8192`, `hq=8`, `hkv=1`), covering:
      - `chunks=(512, 1024)` (`T=24K`),
      - non-24K `(384, 576, 576)` (`c_k % 64 == 0`), and
      - short single-chunk bucket (`t_shard=64`).
    """
    jax = pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    from jax.experimental import topologies
    from jax.sharding import Mesh

    if len(jax.devices("cpu")) < _TP:
        pytest.skip(f"Need {_TP} CPU devices (got {len(jax.devices('cpu'))})")

    with (
        mock.patch.object(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_VARIANT, "get", return_value="v10_sp"
        ),
        mock.patch.object(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_A3_VARIANT, "get", return_value="v16"
        ),
        mock.patch.object(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM, "get", return_value=True
        ),
        mock.patch.object(
            tpu_envs.SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_SPEC,
            "get",
            return_value="1536:512,1024",
        ),
    ):
        from sglang_tpu.muse_prefill_kernels.integration import _ops

        topo = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x=2x2x2")
        mesh = Mesh(np.array(topo.devices).reshape(_TP), (_AXIS,))
        _ops.cache_clear()
        with mock.patch(
            "sglang_tpu.moe.pallas.fused_moe_rs_op._get_mesh", return_value=mesh
        ):
            ops = _ops(
                8,
                1,
                "v10_sp",
                "v16",
                True,
                1e-5,
                a1_rpa_layout=True,
                causal_chunk_stream=True,
                causal_chunk_spec="1536:512,1024",
            )

        assert ops.a1_ulysses_qkv_chunk is not None
        assert ops.a1_ulysses_a2a_pack_chunk is not None
        assert ops.a1_ulysses_gate_chunk is not None
        assert ops.a3_ulysses_rev_a2a_chunk is not None
        assert ops.a3_ulysses_gate_fuse_chunk is not None
        assert ops.a3_ulysses_wo is not None
        assert ops.a3_ulysses_wo_2chunk is not None
        assert ops.moe_agd is None
        assert ops.ag_chunk is None

        h_dim, q_loc = 8192, 512

        for c_k in (512, 1024, 576, 64):
            c_tok = _TP * c_k
            x_t = torch.empty((c_k, h_dim), dtype=torch.bfloat16, device="meta")
            wq_t = torch.empty((_TP, h_dim, 640), dtype=torch.float8_e4m3fn, device="meta")
            wqs_t = torch.empty((_TP, 1, 640), dtype=torch.float32, device="meta")
            cos_t = torch.empty((c_tok, 128), dtype=torch.float32, device="meta")
            sin_t = torch.empty((c_tok, 128), dtype=torch.float32, device="meta")
            _assert_tpu7x_spmd_export_and_compile(ops.a1_ulysses_qkv_chunk, (x_t, wq_t, wqs_t, cos_t, sin_t))

            q3_t = torch.empty((_TP, c_k, q_loc), dtype=torch.bfloat16, device="meta")
            kv3_t = torch.empty((_TP, c_k, 128), dtype=torch.float8_e4m3fn, device="meta")
            _assert_tpu7x_spmd_export_and_compile(ops.a1_ulysses_a2a_pack_chunk, (q3_t, kv3_t))

            xf8_t = torch.empty((c_k, h_dim), dtype=torch.float8_e4m3fn, device="meta")
            wg_t = torch.empty((h_dim, h_dim), dtype=torch.float8_e4m3fn, device="meta")
            wgs_t = torch.empty((1, h_dim), dtype=torch.float32, device="meta")
            _assert_tpu7x_spmd_export_and_compile(ops.a1_ulysses_gate_chunk, (xf8_t, wg_t, wgs_t))

            raw_out_t = torch.empty((c_tok, q_loc), dtype=torch.bfloat16, device="meta")
            _assert_tpu7x_spmd_export_and_compile(ops.a3_ulysses_rev_a2a_chunk, (raw_out_t,))

            af8_t = torch.empty((_TP, c_k, q_loc), dtype=torch.float8_e4m3fn, device="meta")
            gk_t = torch.empty((c_k, h_dim), dtype=torch.bfloat16, device="meta")
            _assert_tpu7x_spmd_export_and_compile(ops.a3_ulysses_gate_fuse_chunk, (af8_t, gk_t))

            qf8_t = torch.empty((c_k, h_dim), dtype=torch.float8_e4m3fn, device="meta")
            wo_t = torch.empty((h_dim, h_dim), dtype=torch.float8_e4m3fn, device="meta")
            wos_t = torch.empty((1, h_dim), dtype=torch.float32, device="meta")
            _assert_tpu7x_spmd_export_and_compile(ops.a3_ulysses_wo, (qf8_t, wo_t, wos_t))

        q0_t = torch.empty((512, h_dim), dtype=torch.float8_e4m3fn, device="meta")
        q1_t = torch.empty((1024, h_dim), dtype=torch.float8_e4m3fn, device="meta")
        _assert_tpu7x_spmd_export_and_compile(ops.a3_ulysses_wo_2chunk, (q0_t, q1_t, wo_t, wos_t))


def test_v10_sp_weights_gates_use_agd_false_and_l61_prune():
    """Verify `_LayerWeights` (replicated QKV+Gate, unpermuted Wo), opt-in gates,
    `use_agd_out=False`, and `_l61_pruned_ulysses_v10_attn_inputs` on CCS + single-chunk.
    """
    torch = pytest.importorskip("torch")
    from sglang_tpu.muse_prefill_kernels.integration import (
        MusePrefillKernels,
        _LayerWeights,
        _prepare_wo,
    )

    h_dim, q_loc, kv_loc = 256, 512, 64
    torch.manual_seed(123)

    shards_o = [torch.randn((h_dim, q_loc), dtype=torch.bfloat16) for _ in range(_TP)]

    def _mock_gather(t):
        return torch.cat([s.t().contiguous() for s in shards_o], dim=0)

    wo_seq, _, shape_seq = _prepare_wo(
        shards_o[0], a3_variant="v16", gather=_mock_gather, tp_rank=3, skip_v16_perm=True
    )
    _, _, shape_v16 = _prepare_wo(
        shards_o[0], a3_variant="v16", gather=_mock_gather, tp_rank=3, skip_v16_perm=False
    )
    assert shape_seq == (_TP * q_loc, h_dim)
    assert shape_v16 == (_TP, q_loc, h_dim)
    assert wo_seq.shape == (_TP * q_loc, h_dim)

    class _Norm:
        def __init__(self):
            self.weight = torch.zeros(h_dim, dtype=torch.bfloat16)

    class _Gate:
        def __init__(self, w):
            self.weight = w

    class _Attn:
        def __init__(self, qkvw, gw, ow):
            self.q_size = q_loc
            self.kv_size = kv_loc
            self.qkv_proj = _Gate(qkvw)
            self.gate_proj = _Gate(gw)
            self.o_proj = _Gate(ow)

    class _Experts:
        w13_weight = torch.zeros(1)

    class _Mlp:
        def __init__(self):
            self.experts = _Experts()
            self.pre_expert_proj = _Gate(torch.randn((16, h_dim), dtype=torch.bfloat16))
            self.post_expert_proj = _Gate(torch.randn((h_dim, 16), dtype=torch.bfloat16))
            self.gate = _Gate(torch.randn((16, h_dim), dtype=torch.float32))

    class _Layer:
        def __init__(self):
            self.self_attn = _Attn(
                torch.randn((q_loc + 2 * kv_loc, h_dim), dtype=torch.bfloat16),
                torch.randn((q_loc, h_dim), dtype=torch.bfloat16),
                shards_o[0],
            )
            self.mlp = _Mlp()
            self.input_layernorm = _Norm()

    layer = _Layer()

    def _uniform_gather(t):
        if t.shape == (q_loc, h_dim) and t.dtype == torch.bfloat16:
            return torch.cat([s.t().contiguous() for s in shards_o], dim=0)
        return torch.cat([t] * _TP, dim=0)

    lw = _LayerWeights(layer, _uniform_gather, "v16", tp_rank=0, a1_variant="v10_sp")
    assert lw.wqkvg.shape == (_TP, h_dim, q_loc + 2 * kv_loc)
    assert lw.wqkvg_scale.shape == (_TP, q_loc + 2 * kv_loc)
    assert lw.w_gate.shape == (h_dim, _TP * q_loc)
    assert lw.w_gate_scale.shape == (_TP * q_loc,)
    assert lw.wo_shape == (_TP * q_loc, h_dim)
    assert layer.self_attn.qkv_proj.weight.shape == (q_loc + 2 * kv_loc, h_dim)
    assert layer.self_attn.gate_proj.weight.shape == (q_loc, h_dim)

    class _FakeOps:
        def a1_ulysses_qkv_chunk(self, x_k, wq, sc, c_k, s_k):
            q3 = torch.zeros((_TP, x_k.shape[0], q_loc), dtype=torch.bfloat16)
            kv3 = torch.zeros((_TP, x_k.shape[0], 128), dtype=torch.float8_e4m3fn)
            xf8 = torch.zeros((x_k.shape[0], h_dim), dtype=torch.float8_e4m3fn)
            return q3, kv3, xf8

        def a1_ulysses_a2a_pack_chunk(self, q3, kv3):
            c_tok = _TP * q3.shape[1]
            return (
                torch.ones((1, c_tok, 4, 2, 128), dtype=torch.bfloat16),
                torch.ones((c_tok, 1, 4, 128), dtype=torch.float8_e4m3fn),
            )

        def a1_ulysses_gate_chunk(self, x_k, wg, wgs):
            return torch.arange(x_k.shape[0], dtype=torch.bfloat16)[:, None].expand(
                -1, _TP * q_loc
            ).contiguous()

        def sum_stream(self, rows):
            return rows + 1.0

    class _FakeBe:
        def __init__(self):
            self.kv_writes = 0

        def write_kv_only(self, kv, attn_l, fb, chunk_meta=None):
            self.kv_writes += 1

    fake_pk = MusePrefillKernels.__new__(MusePrefillKernels)
    fake_pk.ops = _FakeOps()
    fake_pk.tp_size = _TP
    fake_pk.tp_rank = 2
    be = _FakeBe()
    fb = mock.MagicMock(extend_seq_lens=torch.tensor([5, 0], dtype=torch.int32))
    idx_128 = torch.zeros((128,), dtype=torch.long)
    eff_idx = torch.tensor([0, 0], dtype=torch.long)
    active = torch.tensor([True, False])

    qp, gp, kvp = fake_pk._l61_pruned_ulysses_v10_attn_inputs(
        weights=lw,
        attn_in=torch.zeros((64, h_dim), dtype=torch.bfloat16),
        rope=(
            torch.zeros((_TP * 64, 128), dtype=torch.float32),
            torch.zeros((_TP * 64, 128), dtype=torch.float32),
        ),
        chunks=None,
        rope_chunks=None,
        chunk_metas=None,
        forward_batch=fb,
        attn=mock.MagicMock(attn="mock"),
        be=be,
        idx_128=idx_128,
        eff_idx=eff_idx,
        active=active,
        bs=2,
        q_size=q_loc,
        t_shard=64,
        inv_cm_perm=None,
    )
    assert be.kv_writes == 1
    assert qp.shape == (1, 128, 4, 2, 128)
    assert gp.shape == (128, q_loc)
    assert kvp.shape == (128, 1, 4, 128)

    # Also verify multi-chunk CCS routing inside _l61_pruned_ulysses_v10_attn_inputs.
    be2 = _FakeBe()
    qp2, gp2, kvp2 = fake_pk._l61_pruned_ulysses_v10_attn_inputs(
        weights=lw,
        attn_in=torch.zeros((48, h_dim), dtype=torch.bfloat16),
        rope=(
            torch.zeros((_TP * 48, 128), dtype=torch.float32),
            torch.zeros((_TP * 48, 128), dtype=torch.float32),
        ),
        chunks=(16, 32),
        rope_chunks=(
            (torch.zeros((_TP * 16, 128)), torch.zeros((_TP * 16, 128))),
            (torch.zeros((_TP * 32, 128)), torch.zeros((_TP * 32, 128))),
        ),
        chunk_metas=("m0", "m1"),
        forward_batch=fb,
        attn=mock.MagicMock(attn="mock"),
        be=be2,
        idx_128=idx_128,
        eff_idx=eff_idx,
        active=active,
        bs=2,
        q_size=q_loc,
        t_shard=48,
        inv_cm_perm=torch.arange(16, dtype=torch.long),
    )
    assert be2.kv_writes == 2
    assert qp2.shape == (1, 128, 4, 2, 128)
    assert gp2.shape == (128, q_loc)
    assert kvp2.shape == (128, 1, 4, 128)


def test_v10_sp_tp8_interpret_nan_parity_vs_v6_and_rev_bf16():
    """8-device CPU interpret parity (`uninitialized_memory='nan'`) for TP=8 (`hq=16, hkv=2`):
    - `pallas_a1_ulysses_qkv_local` (`sub_n=128`) + 2-op A2A + `pallas_pack_ulysses_qkv_for_rpa`
      matches `v6` (`rpa_layout=True`) bit-for-bit on both `q_5d` and packed `kv_4d`.
    - `pallas_ulysses_wo_local` pinned `3072` (`(1024, 2048)` chunks) matches `v6` reference
      for both FP8 rev-A2A and BF16 rev-A2A (`quantize_fp8=False`, `input_is_prescaled=False`).
    """
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from jax import lax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    tp8 = 8
    if len(jax.devices("cpu")) < tp8:
        pytest.skip(f"Need {tp8} CPU devices (got {len(jax.devices('cpu'))})")

    from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1, kernel_a3
    from sglang_tpu.muse_prefill_kernels.causal_chunk_stream import to_chunk_major_perm

    devs = jax.devices("cpu")[:tp8]
    mesh = Mesh(np.array(devs), (_AXIS,))

    chunks = (1024, 2048)
    t_shard = sum(chunks)  # 3072 -> triggers pinned_3072 Wo path cleanly!
    t_total = tp8 * t_shard  # 24576
    h_dim = 256
    h_q_local, h_kv_local = 16, 2
    q_size, kv_size = h_q_local * 64, h_kv_local * 64  # 1024, 128
    n_full = tp8 * q_size  # 8192

    rng = np.random.default_rng(2027)
    attn_in_np = (rng.standard_normal((t_total, h_dim)) * 0.35).astype(np.float32)
    q_w_np = (rng.standard_normal((tp8, q_size, h_dim)) * 0.2).astype(np.float32)
    k_w_np = (rng.standard_normal((tp8, kv_size, h_dim)) * 0.2).astype(np.float32)
    v_w_np = (rng.standard_normal((tp8, kv_size, h_dim)) * 0.2).astype(np.float32)
    g_w_np = (rng.standard_normal((tp8, q_size, h_dim)) * 0.2).astype(np.float32)
    wo_w_np = (rng.standard_normal((h_dim, tp8, q_size)) * 0.15).astype(np.float32)

    theta = rng.uniform(0.0, 2.0, size=(t_total, 128)).astype(np.float32)
    cos_np, sin_np = np.cos(theta), np.sin(theta)
    perm_cm = np.array(to_chunk_major_perm(tp8, chunks), dtype=np.int64)
    attn_in_shard_np = attn_in_np[perm_cm].reshape(tp8, t_shard, h_dim)

    v6_w, v6_sc, ul_w, ul_sc, wg_parts, wgs_parts = [], [], [], [], [], []
    for r in range(tp8):
        qr, kr, vr, gr = q_w_np[r], k_w_np[r], v_w_np[r], g_w_np[r]
        w6, sc6 = _fp8_k_major_np(np.concatenate([qr, gr, kr, vr], axis=0))
        v6_w.append(w6)
        v6_sc.append(sc6[None, :])
        w_qkv, sc_qkv = _fp8_k_major_np(np.concatenate([qr, kr, vr], axis=0))
        ul_w.append(w_qkv)
        ul_sc.append(sc_qkv)
        wg, wgs = _fp8_k_major_np(gr)
        wg_parts.append(wg)
        wgs_parts.append(wgs)

    w_qkv_3d = np.stack(ul_w, axis=0)
    w_qkv_sc_3d = np.stack(ul_sc, axis=0)[:, None, :]
    wg_full = np.concatenate(wg_parts, axis=1)
    wgs_full = np.concatenate(wgs_parts, axis=0)[None, :]
    wo_full, wo_sc_full = _fp8_k_major_np(wo_w_np.reshape(h_dim, n_full))

    def _put(arr, dt):
        return jax.device_put(jnp.asarray(arr, dtype=dt), NamedSharding(mesh, P(_AXIS)))

    with _interpret_on_cpu(mesh):
        attn_in_d = _put(attn_in_shard_np, jnp.bfloat16)
        v6_w_d = _put(np.stack(v6_w, 0), jnp.float8_e4m3fn)
        v6_sc_d = _put(np.stack(v6_sc, 0), jnp.float32)
        ul_w_d = _put(np.broadcast_to(w_qkv_3d[None], (tp8, *w_qkv_3d.shape)), jnp.float8_e4m3fn)
        ul_sc_d = _put(np.broadcast_to(w_qkv_sc_3d[None], (tp8, *w_qkv_sc_3d.shape)), jnp.float32)
        wg_d = _put(np.broadcast_to(wg_full[None], (tp8, *wg_full.shape)), jnp.float8_e4m3fn)
        wgs_d = _put(np.broadcast_to(wgs_full[None], (tp8, *wgs_full.shape)), jnp.float32)
        wo_d = _put(np.broadcast_to(wo_full[None], (tp8, *wo_full.shape)), jnp.float8_e4m3fn)
        wos_d = _put(np.broadcast_to(wo_sc_full[None, None, :], (tp8, 1, h_dim)), jnp.float32)

        c_offs = (0, tp8 * chunks[0])
        cos_d = [
            _put(np.broadcast_to(cos_np[None, c_offs[k] : c_offs[k] + tp8 * c_k], (tp8, tp8 * c_k, 128)), jnp.float32)
            for k, c_k in enumerate(chunks)
        ]
        sin_d = [
            _put(np.broadcast_to(sin_np[None, c_offs[k] : c_offs[k] + tp8 * c_k], (tp8, tp8 * c_k, 128)), jnp.float32)
            for k, c_k in enumerate(chunks)
        ]

        # Test exact Q, K, V, gate parity on chunk 0 (`1024 -> 8192` tokens, `rpa_layout=False` in
        # interpret to avoid JAX Mosaic BitcastTransform) and Wo parity on `(1024, 2048)` (`t_total=3072`).
        def _v6_c0_fn(xk, w6, sc6, cc, ss):
            x_fp8 = jnp.clip(
                xk[0, : chunks[0]].astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
                -448.0,
                448.0,
            ).astype(jnp.float8_e4m3fn)
            ag = lax.all_gather(x_fp8, _AXIS, axis=0, tiled=False)
            q_k, _, kv_k = kernel_a1.pallas_a1_chunk_1way(
                ag,
                w6[0],
                sc6[0] * kernel_a1.STATIC_A1_ACT_SCALE,
                cc[0].reshape(tp8, chunks[0], 128),
                ss[0].reshape(tp8, chunks[0], 128),
                None,
                t_shard=chunks[0],
                row_off=0,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                rpa_layout=False,
                mode="all",
                tp_axis=_AXIS,
                tm=256,
            )
            c_tok = tp8 * chunks[0]
            return q_k.reshape(c_tok, q_size)[None], kv_k.reshape(c_tok, 2 * kv_size)[None]

        def _v10_c0_fn(xk, wul, scul, cc, ss, *, sub_n: int):
            me = lax.axis_index(_AXIS)
            x_fp8 = jnp.clip(
                xk[0, : chunks[0]].astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT,
                -448.0,
                448.0,
            ).astype(jnp.float8_e4m3fn)
            co = lax.dynamic_index_in_dim(cc[0].reshape(tp8, chunks[0], 128), me, axis=0, keepdims=False)
            so = lax.dynamic_index_in_dim(ss[0].reshape(tp8, chunks[0], 128), me, axis=0, keepdims=False)
            q3, kv3 = kernel_a1.pallas_a1_ulysses_qkv_local(
                x_fp8,
                wul[0],
                scul[0] * kernel_a1.STATIC_A1_ACT_SCALE,
                co,
                so,
                h_q_local=h_q_local,
                h_kv_local=h_kv_local,
                sub_n=sub_n,
            )
            c_tok = tp8 * chunks[0]
            q2 = lax.all_to_all(q3, _AXIS, split_axis=0, concat_axis=0, tiled=False).reshape(c_tok, q_size)
            kv2 = lax.all_to_all(kv3, _AXIS, split_axis=0, concat_axis=0, tiled=False).reshape(
                c_tok, 2 * kv_size
            )
            return q2[None], kv2[None]

        sm_v6_c0 = jax.shard_map(_v6_c0_fn, mesh=mesh, in_specs=(P(_AXIS),) * 5, out_specs=(P(_AXIS),) * 2, check_vma=False)
        sm_v10_sn128 = jax.shard_map(lambda *a: _v10_c0_fn(*a, sub_n=128), mesh=mesh, in_specs=(P(_AXIS),) * 5, out_specs=(P(_AXIS),) * 2, check_vma=False)
        sm_v10_sn0 = jax.shard_map(lambda *a: _v10_c0_fn(*a, sub_n=0), mesh=mesh, in_specs=(P(_AXIS),) * 5, out_specs=(P(_AXIS),) * 2, check_vma=False)
        q6_d, kv6_d = jax.block_until_ready(sm_v6_c0(attn_in_d, v6_w_d, v6_sc_d, cos_d[0], sin_d[0]))
        q10_d, kv10_d = jax.block_until_ready(sm_v10_sn128(attn_in_d, ul_w_d, ul_sc_d, cos_d[0], sin_d[0]))
        q10_0_d, kv10_0_d = jax.block_until_ready(sm_v10_sn0(attn_in_d, ul_w_d, ul_sc_d, cos_d[0], sin_d[0]))

        q6_u16 = np.asarray(lax.bitcast_convert_type(q6_d.addressable_shards[0].data[0], jnp.uint16))
        q10_u16 = np.asarray(lax.bitcast_convert_type(q10_d.addressable_shards[0].data[0], jnp.uint16))
        kv6_u8 = np.asarray(lax.bitcast_convert_type(kv6_d.addressable_shards[0].data[0], jnp.uint8))
        kv10_u8 = np.asarray(lax.bitcast_convert_type(kv10_d.addressable_shards[0].data[0], jnp.uint8))
        np.testing.assert_array_equal(q10_u16, q6_u16)
        np.testing.assert_array_equal(kv10_u8, kv6_u8)
        q6_f = np.asarray(q6_d.addressable_shards[0].data[0], dtype=np.float32)
        q0_f = np.asarray(q10_0_d.addressable_shards[0].data[0], dtype=np.float32)
        cos_sn0 = float(np.sum(q6_f * q0_f) / (np.linalg.norm(q6_f) * np.linalg.norm(q0_f) + 1e-12))
        assert cos_sn0 >= 0.9998 and np.all(np.isfinite(q0_f))
        np.testing.assert_array_equal(
            np.asarray(lax.bitcast_convert_type(kv10_0_d.addressable_shards[0].data[0], jnp.uint8)),
            kv6_u8,
        )

        # Verify FP8 rev-A2A, BF16 rev-A2A, and pinned `3072` `pallas_ulysses_wo_local((qp0, qp1), ...)`.
        ar0_np = (rng.standard_normal((tp8, tp8 * chunks[0], q_size)) * 0.4).astype(np.float32)
        ar1_np = (rng.standard_normal((tp8, tp8 * chunks[1], q_size)) * 0.4).astype(np.float32)
        ar0_d, ar1_d = _put(ar0_np, jnp.bfloat16), _put(ar1_np, jnp.bfloat16)

        def _v6_wo_fn(xk, ar0, ar1, w6, sc6, woseq, wosc):
            def _one(x_part, ar_part):
                x_f = jnp.clip(
                    x_part.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT, -448.0, 448.0
                ).astype(jnp.float8_e4m3fn)
                ag = lax.all_gather(x_f, _AXIS, axis=0, tiled=False)
                # Extract gate exact [tp*c_k, q_size] via pallas_a1_chunk_1way.
                c_k = x_part.shape[0]
                t_g = jnp.zeros((tp8, c_k, 128), dtype=jnp.float32)
                _, g_k, _ = kernel_a1.pallas_a1_chunk_1way(
                    ag, w6[0], sc6[0] * kernel_a1.STATIC_A1_ACT_SCALE, t_g, t_g, None,
                    t_shard=c_k, row_off=0, h_q_local=h_q_local, h_kv_local=h_kv_local,
                    rpa_layout=False, mode="all", tp_axis=_AXIS, tm=256,
                )
                qp = kernel_a3.pallas_gate_static_fp8_quant(
                    ar_part, g_k.reshape(tp8 * c_k, q_size), tm=512, head_norm_eps=1e-5, rpa_packed_out=False
                ).reshape(tp8, c_k, q_size)
                return lax.all_to_all(qp, _AXIS, split_axis=0, concat_axis=0, tiled=False).reshape(tp8, c_k, q_size)

            qa0 = _one(xk[0, : chunks[0]], ar0[0])
            qa1 = _one(xk[0, chunks[0] :], ar1[0])
            qa = jnp.concatenate([qa0, qa1], axis=1)
            return (
                jnp.einsum(
                    "rth,rhn->tn",
                    qa.astype(jnp.float32),
                    woseq[0].reshape(tp8, q_size, h_dim).astype(jnp.float32),
                )
                * (wosc[0] * kernel_a3.STATIC_A3_ACT_SCALE)
            ).astype(jnp.bfloat16)[None]

        def _v10_wo_fn(xk, ar0, ar1, wg, wgs, woseq, wosc, *, rev_bf16: bool):
            def _one(x_part, ar_part):
                c_k = x_part.shape[0]
                x_f = jnp.clip(
                    x_part.astype(jnp.float32) * kernel_a1.STATIC_A1_ACT_MULT, -448.0, 448.0
                ).astype(jnp.float8_e4m3fn)
                gk = kernel_a1.pallas_ulysses_local_gate(
                    x_f, wg[0], wgs[0] * kernel_a1.STATIC_A1_ACT_SCALE
                )
                rn = kernel_a3.pallas_ulysses_norm_fp8_for_rev_a2a(
                    ar_part, head_norm_eps=1e-5, quantize_fp8=not rev_bf16
                ).reshape(tp8, c_k, q_size)
                ra = lax.all_to_all(rn, _AXIS, split_axis=0, concat_axis=0, tiled=False).reshape(
                    tp8, c_k, q_size
                )
                return kernel_a3.pallas_ulysses_gate_fuse(
                    ra, gk, input_is_prescaled=not rev_bf16
                )

            qp0 = _one(xk[0, : chunks[0]], ar0[0])
            qp1 = _one(xk[0, chunks[0] :], ar1[0])
            return kernel_a3.pallas_ulysses_wo_local(
                (qp0, qp1), woseq[0], wosc[0], bn=min(1024, h_dim)
            )[None]

        sm_v6_wo = jax.shard_map(_v6_wo_fn, mesh=mesh, in_specs=(P(_AXIS),) * 7, out_specs=P(_AXIS), check_vma=False)
        sm_v10_fp8 = jax.shard_map(lambda *a: _v10_wo_fn(*a, rev_bf16=False), mesh=mesh, in_specs=(P(_AXIS),) * 7, out_specs=P(_AXIS), check_vma=False)
        sm_v10_bf16 = jax.shard_map(lambda *a: _v10_wo_fn(*a, rev_bf16=True), mesh=mesh, in_specs=(P(_AXIS),) * 7, out_specs=P(_AXIS), check_vma=False)

        wo_ref = np.asarray(jax.block_until_ready(sm_v6_wo(attn_in_d, ar0_d, ar1_d, v6_w_d, v6_sc_d, wo_d, wos_d)), dtype=np.float32).reshape(-1, h_dim)
        wo_f8 = np.asarray(jax.block_until_ready(sm_v10_fp8(attn_in_d, ar0_d, ar1_d, wg_d, wgs_d, wo_d, wos_d)), dtype=np.float32).reshape(-1, h_dim)
        wo_bf = np.asarray(jax.block_until_ready(sm_v10_bf16(attn_in_d, ar0_d, ar1_d, wg_d, wgs_d, wo_d, wos_d)), dtype=np.float32).reshape(-1, h_dim)
        assert np.all(np.isfinite(wo_f8)) and np.all(np.isfinite(wo_bf))
        cos_f8 = float(np.sum(wo_ref * wo_f8) / (np.linalg.norm(wo_ref) * np.linalg.norm(wo_f8) + 1e-12))
        cos_bf = float(np.sum(wo_ref * wo_bf) / (np.linalg.norm(wo_ref) * np.linalg.norm(wo_bf) + 1e-12))
        assert cos_f8 >= 0.9985, f"TP=8 FP8 rev-A2A Wo parity cos={cos_f8:.6f} too low"
        assert cos_bf >= 0.9995, f"TP=8 BF16 rev-A2A Wo parity cos={cos_bf:.6f} too low"


def test_v10_sp_tpu7x_2x2x1_tp8_prod_aot_compile_and_rev_bf16():
    """Offline TPU v7 (`2x2x1` TP=8) Mosaic AOT compile + zero-`partition_id` check of ALL
    `v10_sp` ops at full Muse Spark production geometry (`H=8192`, `hq=16`, `hkv=2`), including
    `chunks=(1024, 2048)` (`T=24K`, `pinned_3072` `a3_ulysses_wo_2chunk`), short `c_k=64`,
    and `ULYSSES_REV_BF16=True`.
    """
    jax = pytest.importorskip("jax")
    torch = pytest.importorskip("torch")
    from jax.experimental import topologies
    from jax.sharding import Mesh

    tp8 = 8
    if len(jax.devices("cpu")) < tp8:
        pytest.skip(f"Need {tp8} CPU devices (got {len(jax.devices('cpu'))})")

    from sglang_tpu.muse_prefill_kernels.integration import _ops

    topo = topologies.get_topology_desc(platform="tpu", topology_name="tpu7x=2x2x1")
    mesh = Mesh(np.array(topo.devices).reshape(tp8), (_AXIS,))
    h_dim, q_loc, kv_cols = 8192, 1024, 256

    _ops.cache_clear()
    with mock.patch("sglang_tpu.moe.pallas.fused_moe_rs_op._get_mesh", return_value=mesh):
        ops_f8 = _ops(
            16, 2, "v10_sp", "v16", True, 1e-5,
            a1_rpa_layout=True, causal_chunk_stream=True,
            causal_chunk_spec="3072:1024,2048", ulysses_rev_bf16=False,
        )
        ops_bf16 = _ops(
            16, 2, "v10_sp", "v16", True, 1e-5,
            a1_rpa_layout=True, causal_chunk_stream=True,
            causal_chunk_spec="3072:1024,2048", ulysses_rev_bf16=True,
        )

    assert ops_f8.a3_ulysses_rev_a2a_chunk._init_fn.name != ops_bf16.a3_ulysses_rev_a2a_chunk._init_fn.name
    assert ops_f8.a3_ulysses_gate_fuse_chunk._init_fn.name != ops_bf16.a3_ulysses_gate_fuse_chunk._init_fn.name

    for c_k in (1024, 2048, 64):
        c_tok = tp8 * c_k
        x_t = torch.empty((c_k, h_dim), dtype=torch.bfloat16, device="meta")
        wq_t = torch.empty((tp8, h_dim, q_loc + kv_cols), dtype=torch.float8_e4m3fn, device="meta")
        wqs_t = torch.empty((tp8, 1, q_loc + kv_cols), dtype=torch.float32, device="meta")
        cos_t = torch.empty((c_tok, 128), dtype=torch.float32, device="meta")
        sin_t = torch.empty((c_tok, 128), dtype=torch.float32, device="meta")
        _assert_tpu7x_spmd_export_and_compile(ops_f8.a1_ulysses_qkv_chunk, (x_t, wq_t, wqs_t, cos_t, sin_t))

        q3_t = torch.empty((tp8, c_k, q_loc), dtype=torch.bfloat16, device="meta")
        kv3_t = torch.empty((tp8, c_k, kv_cols), dtype=torch.float8_e4m3fn, device="meta")
        _assert_tpu7x_spmd_export_and_compile(ops_f8.a1_ulysses_a2a_pack_chunk, (q3_t, kv3_t))

        xf8_t = torch.empty((c_k, h_dim), dtype=torch.float8_e4m3fn, device="meta")
        wg_t = torch.empty((h_dim, h_dim), dtype=torch.float8_e4m3fn, device="meta")
        wgs_t = torch.empty((1, h_dim), dtype=torch.float32, device="meta")
        _assert_tpu7x_spmd_export_and_compile(ops_f8.a1_ulysses_gate_chunk, (xf8_t, wg_t, wgs_t))

        raw_out_t = torch.empty((c_tok, q_loc), dtype=torch.bfloat16, device="meta")
        gk_t = torch.empty((c_k, h_dim), dtype=torch.bfloat16, device="meta")
        _assert_tpu7x_spmd_export_and_compile(ops_f8.a3_ulysses_rev_a2a_chunk, (raw_out_t,))
        _assert_tpu7x_spmd_export_and_compile(ops_bf16.a3_ulysses_rev_a2a_chunk, (raw_out_t,))

        _assert_tpu7x_spmd_export_and_compile(
            ops_f8.a3_ulysses_gate_fuse_chunk,
            (torch.empty((tp8, c_k, q_loc), dtype=torch.float8_e4m3fn, device="meta"), gk_t),
        )
        _assert_tpu7x_spmd_export_and_compile(
            ops_bf16.a3_ulysses_gate_fuse_chunk,
            (torch.empty((tp8, c_k, q_loc), dtype=torch.bfloat16, device="meta"), gk_t),
        )

    wo_t = torch.empty((h_dim, h_dim), dtype=torch.float8_e4m3fn, device="meta")
    wos_t = torch.empty((1, h_dim), dtype=torch.float32, device="meta")
    _assert_tpu7x_spmd_export_and_compile(
        ops_f8.a3_ulysses_wo_2chunk,
        (
            torch.empty((1024, h_dim), dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty((2048, h_dim), dtype=torch.float8_e4m3fn, device="meta"),
            wo_t,
            wos_t,
        ),
    )


def test_v10_sp_from_model_validation_tp8_and_tp16():
    """Verify `MusePrefillKernels.from_model` cleanly validates TP=8 & TP=16 `v10_sp` flags."""
    import os
    import types

    torch = pytest.importorskip("torch")
    from sglang_tpu.muse_prefill_kernels import integration
    from sglang_tpu.muse_prefill_kernels.attn_kernels.reference import QUERY_SCALE

    cfg = types.SimpleNamespace(
        num_attention_heads=128,
        num_key_value_heads=16,
        qk_scale_factor=QUERY_SCALE,
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
    fake_decoder = types.SimpleNamespace(
        pp_group=types.SimpleNamespace(world_size=1),
        layers=[
            types.SimpleNamespace(
                mlp=types.SimpleNamespace(
                    experts=types.SimpleNamespace(w13_weight=torch.empty(0))
                )
            )
        ],
    )
    fake_model = types.SimpleNamespace(model=fake_decoder, text_config=cfg)

    def _run_from_model(tp_sz, extra_env, *, ep_tp2=False):
        par = types.SimpleNamespace(
            tp_size=tp_sz,
            attn_tp_size=tp_sz,
            moe_ep_size=(tp_sz // 2) if ep_tp2 else tp_sz,
            moe_tp_size=2 if ep_tp2 else 1,
            attn_dcp_size=1,
        )
        base_env = {
            "SGLANG_TPU_MUSE_KERNEL_A1_VARIANT": "v10_sp",
            "SGLANG_TPU_MUSE_KERNEL_A3_VARIANT": "v16",
            "SGLANG_TPU_MUSE_KERNEL_CAUSAL_CHUNK_STREAM": "1",
            "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "1",
            "SGLANG_TPU_MUSE_KERNEL_FORCE": "0",
        }
        base_env.update(extra_env)
        with (
            mock.patch("sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False),
            mock.patch("sglang.srt.runtime_context.get_parallel", return_value=par),
            mock.patch.dict(os.environ, base_env),
            mock.patch.object(integration.logger, "warning") as mock_warn,
        ):
            integration.MusePrefillKernels.from_model(fake_model)
            return [str(call.args[1]) for call in mock_warn.call_args_list if len(call.args) > 1]

    # TP=8 valid: KV_PACK_HEADS=1 -> passes _v10_sp_config_error and reaches layer check.
    r_tp8_ok = _run_from_model(8, {"SGLANG_TPU_KV_PACK_HEADS": "1"})
    assert not any("v10_sp" in r for r in r_tp8_ok), r_tp8_ok

    # TP=8 invalid: missing KV_PACK_HEADS, missing A1_RPA_LAYOUT, or ep_tp2=1.
    for bad_env in (
        {"SGLANG_TPU_KV_PACK_HEADS": "0"},
        {"SGLANG_TPU_KV_PACK_HEADS": "1", "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "0"},
        {"SGLANG_TPU_KV_PACK_HEADS": "1", "SGLANG_TPU_MUSE_KERNEL_A3_VARIANT": "v3"},
    ):
        reasons = _run_from_model(8, bad_env)
        assert any("v10_sp" in r for r in reasons), (bad_env, reasons)
    reasons_ep2 = _run_from_model(8, {"SGLANG_TPU_KV_PACK_HEADS": "1", "SGLANG_TPU_MUSE_KERNEL_EP_TP2_PREFILL": "1"}, ep_tp2=True)
    assert any("v10_sp" in r for r in reasons_ep2), reasons_ep2

    # TP=16 valid: TP16_PREFILL=1, KV_PACK_HEADS=0, A1_RPA_LAYOUT=1.
    r_tp16_ok = _run_from_model(16, {"SGLANG_TPU_MUSE_KERNEL_TP16_PREFILL": "1", "SGLANG_TPU_KV_PACK_HEADS": "0"})
    assert not any("v10_sp" in r for r in r_tp16_ok), r_tp16_ok

    # TP=16 invalid: missing A1_RPA_LAYOUT=1 or KV_PACK_HEADS=1.
    for bad_env in (
        {"SGLANG_TPU_MUSE_KERNEL_TP16_PREFILL": "1", "SGLANG_TPU_KV_PACK_HEADS": "0", "SGLANG_TPU_MUSE_KERNEL_A1_RPA_LAYOUT": "0"},
        {"SGLANG_TPU_MUSE_KERNEL_TP16_PREFILL": "1", "SGLANG_TPU_KV_PACK_HEADS": "1"},
    ):
        reasons = _run_from_model(16, bad_env)
        assert any("v10_sp" in r for r in reasons), (bad_env, reasons)


def test_v10_sp_code_hygiene_ast_and_defaults():
    """Verify default flags stay unchanged, AST parses cleanly, and 0 raw os.environ.get."""
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_A1_VARIANT.get() == "v1"
    assert tpu_envs.SGLANG_TPU_MUSE_KERNEL_ULYSSES_REV_BF16.get() is False
    repo = pathlib.Path(__file__).resolve().parents[5]
    for rel in (
        "sglang_tpu/envs.py",
        "sglang_tpu/muse_prefill_kernels/attn_kernels/kernel_a1.py",
        "sglang_tpu/muse_prefill_kernels/attn_kernels/kernel_a3.py",
        "sglang_tpu/muse_prefill_kernels/integration.py",
    ):
        src = (repo / rel).read_text()
        ast.parse(src)
        assert "os.environ.get(" not in src, f"Raw os.environ.get found in {rel}"
