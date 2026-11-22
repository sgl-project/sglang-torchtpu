# SPDX-License-Identifier: Apache-2.0
"""Unit, CPU interpret-mode race/NaN, and TPU7x Mosaic export tests for Wave-2 Attn/RPA/Prefill."""

import os
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np


def _fp8_from_u8(arr_u8: np.ndarray) -> jax.Array:
    return jax.lax.bitcast_convert_type(jnp.asarray(arr_u8, dtype=jnp.uint8), jnp.float8_e4m3fn)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def _exact_decode_ref(
    q_3d: np.ndarray,
    kv_merged: np.ndarray,
    kv_cache_u8: np.ndarray,
    kv_lens: np.ndarray,
    page_indices: np.ndarray,
    cu_q_lens: np.ndarray,
    cu_kv_lens: np.ndarray,
    n_active: int,
    *,
    sm_scale: float,
    sliding_window: int | None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    dq: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact FP32 causal/SWA reference for packed 16Q/2KV (16x128) decode & verify."""
    page_size = kv_cache_u8.shape[1]
    cache_after = kv_cache_u8.copy()
    out = np.zeros_like(q_3d, dtype=np.float32)
    kv_u8 = np.asarray(
        jax.lax.bitcast_convert_type(jnp.asarray(kv_merged), jnp.uint8)
    )
    for seq in range(n_active):
        q0 = int(cu_q_lens[seq])
        q1 = int(cu_q_lens[seq + 1])
        kvl = int(kv_lens[seq])
        if (q1 - q0) != dq or kvl < dq:
            continue
        pg_base = (int(cu_kv_lens[seq]) + page_size - 1) // page_size
        for j in range(dq):
            pos = kvl - dq + j
            dst_pg = int(page_indices[pg_base + pos // page_size])
            cache_after[dst_pg, pos % page_size] = kv_u8[q0 + j, 0]
        # Gather [0 .. kvl - 1] from cache_after
        k_rows = []
        v_rows = []
        for pos in range(kvl):
            pg = int(page_indices[pg_base + pos // page_size])
            row_u8 = cache_after[pg, pos % page_size, 0]  # [4, 128]
            k_f = np.asarray(_fp8_from_u8(row_u8[0]).astype(jnp.float32)) * k_scale
            v_f = np.asarray(_fp8_from_u8(row_u8[1]).astype(jnp.float32)) * v_scale
            k_rows.append(k_f)
            v_rows.append(v_f)
        k_mat = np.stack(k_rows, axis=0)  # [kvl, 128]
        v_mat = np.stack(v_rows, axis=0)  # [kvl, 128]
        for j in range(dq):
            q_pos = kvl - dq + j
            lo = 0 if sliding_window is None else max(0, q_pos - (sliding_window - 1))
            hi = q_pos + 1
            q_f = q_3d[q0 + j].astype(np.float32)  # [16, 128]
            scores = (q_f @ k_mat[lo:hi].T) * sm_scale
            scores = scores - np.max(scores, axis=-1, keepdims=True)
            probs = np.exp(scores)
            probs = probs / np.sum(probs, axis=-1, keepdims=True)
            out[q0 + j] = probs @ v_mat[lo:hi]
    return out, cache_after


class TestMuseWave2AttnRpa(unittest.TestCase):
    def test_default_off_flags_preserve_stock_behavior(self):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            can_use_decode_fast,
            parse_decode_fast_config,
        )
        from sglang_tpu.envs import envs

        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_EP_TP2_PREFILL.get())
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_FINER_PREFILL_BUCKETS.get())
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_DECODE_NORM_FAST.get())
        self.assertFalse(envs.SGLANG_TPU_MUSE_DECODE_ATTN_GATE_FUSE.get())
        self.assertFalse(envs.SGLANG_TPU_RPA_V3_DECODE_FAST_V2.get())
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_A3_ZERO_SPILL_PROLOGUE.get())
        self.assertEqual(envs.SGLANG_TPU_BRIDGE_ALL_REDUCE_MAX_KB.get(), 0)

        # AST guard: ensure no sglang_tpu file evaluates getattr(..., "SGLANG_...", None)
        # or envs.SGLANG_* directly in boolean context (EnvVar.__bool__ raises RuntimeError).
        import ast
        import pathlib

        pkg_root = pathlib.Path(__file__).resolve().parents[5] / "sglang_tpu"
        for py_path in pkg_root.rglob("*.py"):
            tree = ast.parse(py_path.read_text(), filename=str(py_path))
            for node in ast.walk(tree):
                exprs = []
                if isinstance(node, ast.BoolOp):
                    exprs.extend(node.values)
                elif isinstance(node, (ast.If, ast.IfExp, ast.While)):
                    exprs.append(node.test)
                elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                    exprs.append(node.operand)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "bool"
                    and node.args
                ):
                    exprs.append(node.args[0])
                for ex in exprs:
                    bad_getattr = (
                        isinstance(ex, ast.Call)
                        and isinstance(ex.func, ast.Name)
                        and ex.func.id == "getattr"
                        and len(ex.args) >= 2
                        and isinstance(ex.args[1], ast.Constant)
                        and isinstance(ex.args[1].value, str)
                        and ex.args[1].value.startswith("SGLANG_")
                    )
                    bad_attr = (
                        isinstance(ex, ast.Attribute)
                        and isinstance(ex.attr, str)
                        and ex.attr.startswith("SGLANG_")
                    )
                    self.assertFalse(
                        bad_getattr or bad_attr,
                        f"Direct EnvVar boolean check at {py_path}:{ex.lineno}: {ast.unparse(ex)}",
                    )

        q = jnp.zeros((16, 16, 128), dtype=jnp.bfloat16)
        kv_cache = jnp.zeros((256, 128, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        # Stock v1 rejects pps=129 and dq=2
        self.assertFalse(
            can_use_decode_fast(
                mode_str="1",
                q=q,
                kv_cache_fused=kv_cache,
                out_dtype=jnp.bfloat16,
                decode_query_size=1,
                custom_mask=None,
                attention_sink=None,
                soft_cap=None,
                xai_temperature_len=None,
                q_scale=None,
                actual_num_q_heads=16,
                actual_num_kv_heads=1,
                actual_head_dim=128,
                sliding_window=None,
                pages_per_seq=129,
            )
        )
        cfg_v1 = parse_decode_fast_config("1", sliding_window=2049, pages_per_seq=129)
        self.assertEqual(cfg_v1.loop_kind, "stream0b")
        self.assertFalse(cfg_v1.coalesce_dma)
        self.assertFalse(cfg_v1.in_vmem_pack2)

        with patch.dict(os.environ, {"SGLANG_TPU_RPA_V3_DECODE_FAST_V2": "1"}):
            # Default v2 keeps Full pps=129 on stock RPAd-zt (0 regression at kv=8192)
            # while explicit streamdyn unlocks pps<=144 dynamic streaming for short kv!
            self.assertFalse(
                can_use_decode_fast(
                    mode_str="1",
                    q=q,
                    kv_cache_fused=kv_cache,
                    out_dtype=jnp.bfloat16,
                    decode_query_size=1,
                    custom_mask=None,
                    attention_sink=None,
                    soft_cap=None,
                    xai_temperature_len=None,
                    q_scale=None,
                    actual_num_q_heads=16,
                    actual_num_kv_heads=1,
                    actual_head_dim=128,
                    sliding_window=None,
                    pages_per_seq=129,
                )
            )
            self.assertTrue(
                can_use_decode_fast(
                    mode_str="streamdyn",
                    q=q,
                    kv_cache_fused=kv_cache,
                    out_dtype=jnp.bfloat16,
                    decode_query_size=1,
                    custom_mask=None,
                    attention_sink=None,
                    soft_cap=None,
                    xai_temperature_len=None,
                    q_scale=None,
                    actual_num_q_heads=16,
                    actual_num_kv_heads=1,
                    actual_head_dim=128,
                    sliding_window=None,
                    pages_per_seq=129,
                )
            )
            # Default v2 keeps dq > 1 routed to VERIFY_RPA_FAST=1 (RPAd-zt 67/77 us)
            # unless explicit mode_str contains "multiq".
            self.assertFalse(
                can_use_decode_fast(
                    mode_str="1",
                    q=q,
                    kv_cache_fused=kv_cache,
                    out_dtype=jnp.bfloat16,
                    decode_query_size=4,
                    custom_mask=None,
                    attention_sink=None,
                    soft_cap=None,
                    xai_temperature_len=None,
                    q_scale=None,
                    actual_num_q_heads=16,
                    actual_num_kv_heads=1,
                    actual_head_dim=128,
                    sliding_window=2049,
                    pages_per_seq=129,
                )
            )
            self.assertTrue(
                can_use_decode_fast(
                    mode_str="multiq",
                    q=q,
                    kv_cache_fused=kv_cache,
                    out_dtype=jnp.bfloat16,
                    decode_query_size=4,
                    custom_mask=None,
                    attention_sink=None,
                    soft_cap=None,
                    xai_temperature_len=None,
                    q_scale=None,
                    actual_num_q_heads=16,
                    actual_num_kv_heads=1,
                    actual_head_dim=128,
                    sliding_window=2049,
                    pages_per_seq=129,
                )
            )
            cfg_full = parse_decode_fast_config("streamdyn", sliding_window=None, pages_per_seq=129)
            self.assertEqual(cfg_full.loop_kind, "streamdyn")
            self.assertEqual(cfg_full.chunk_sz, 2048)
            self.assertEqual(cfg_full.max_ring_pages, 144)
            self.assertFalse(cfg_full.coalesce_dma)
            self.assertTrue(cfg_full.in_vmem_pack2)
            # Explicit streamdyn unlocks Full-attn pps<=144 without demoting SWA (win_pages<=32) off stream0b!
            cfg_sw_sdyn = parse_decode_fast_config(
                "streamdyn", sliding_window=2048, pages_per_seq=129
            )
            self.assertEqual(cfg_sw_sdyn.loop_kind, "stream0b")
            self.assertEqual(cfg_sw_sdyn.max_ring_pages, 17)
            cfg_sw2049_sdyn = parse_decode_fast_config(
                "streamdyn", sliding_window=2049, pages_per_seq=129
            )
            self.assertEqual(cfg_sw2049_sdyn.loop_kind, "stream0b")
            self.assertEqual(cfg_sw2049_sdyn.max_ring_pages, 17)
            cfg_sw = parse_decode_fast_config("1", sliding_window=2049, pages_per_seq=129)
            self.assertEqual(cfg_sw.loop_kind, "stream0b")
            cfg_sw_dq2 = parse_decode_fast_config(
                "1", sliding_window=2049, pages_per_seq=129, decode_query_size=2
            )
            self.assertEqual(cfg_sw_dq2.loop_kind, "stream0b")
            cfg_sw_dq4 = parse_decode_fast_config(
                "1", sliding_window=2049, pages_per_seq=129, decode_query_size=4
            )
            self.assertEqual(cfg_sw_dq4.loop_kind, "stream0b")

    def test_d3v2_cpu_interpret_nan_safety_and_multi_q_verify(self):
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        rng = np.random.default_rng(1119)
        page_size = 128
        for sliding_window, pps, kv_list, dq in [
            (2049, 24, [2500, 1, 128, 2049, 2050, 45, 512, 17], 1),
            (None, 24, [1500, 1, 128, 129, 1024, 1025, 2300, 19], 1),
            (2049, 24, [2400, 4, 2050, 513], 2),
            (None, 20, [1400, 4, 1026, 259], 4),
        ]:
            b_seqs = len(kv_list)
            b_tok = max(8, b_seqs * dq)
            # Fill entire KV cache with 0x7F FP8 NaN first!
            total_pages = b_seqs * pps + 8
            cache_u8 = np.full((total_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
            page_indices = np.arange(b_seqs * pps, dtype=np.int32)
            cu_q = np.zeros(b_seqs + 1, dtype=np.int32)
            cu_kv = np.zeros(b_seqs + 1, dtype=np.int32)
            for s, kvl in enumerate(kv_list):
                cu_q[s + 1] = cu_q[s] + dq
                cu_kv[s + 1] = (s + 1) * pps * page_size
                pg_base = s * pps
                cache_kv = kvl - dq
                lo_valid = (
                    0
                    if sliding_window is None
                    else max(0, cache_kv - (sliding_window - 1))
                )
                # Populate ONLY valid prefix [lo_valid, cache_kv) with finite FP8;
                # leave < lo_valid (evicted SWA prefix) and >= cache_kv as 0x7F NaN!
                for pos in range(lo_valid, cache_kv):
                    pg = page_indices[pg_base + pos // page_size]
                    fin_k = np.asarray(
                        jnp.asarray(rng.standard_normal(128) * 0.5, dtype=jnp.float8_e4m3fn).view(jnp.uint8)
                    )
                    fin_v = np.asarray(
                        jnp.asarray(rng.standard_normal(128) * 0.5, dtype=jnp.float8_e4m3fn).view(jnp.uint8)
                    )
                    cache_u8[pg, pos % page_size, 0, 0] = fin_k
                    cache_u8[pg, pos % page_size, 0, 1] = fin_v
                    cache_u8[pg, pos % page_size, 0, 2:] = 0

            q_np = (rng.standard_normal((b_tok, 16, 128)) * 0.4).astype(np.float32)
            q_j = jnp.asarray(q_np, dtype=jnp.bfloat16)
            q_5d = q_j.reshape(1, b_tok, 8, 2, 128)
            b_pad32 = ((b_tok + 31) // 32) * 32
            kv_new_u8 = np.zeros((b_pad32, 1, 4, 128), dtype=np.uint8)
            kv_new_u8[:b_tok, 0, 0] = np.asarray(
                jnp.asarray(rng.standard_normal((b_tok, 128)) * 0.5, dtype=jnp.float8_e4m3fn).view(jnp.uint8)
            )
            kv_new_u8[:b_tok, 0, 1] = np.asarray(
                jnp.asarray(rng.standard_normal((b_tok, 128)) * 0.5, dtype=jnp.float8_e4m3fn).view(jnp.uint8)
            )
            kv_merged = _fp8_from_u8(kv_new_u8)
            gate_j = jnp.asarray(rng.standard_normal((b_tok, 1024)), dtype=jnp.bfloat16)
            kv_lens_j = jnp.asarray(kv_list, dtype=jnp.int32)
            dist_j = jnp.asarray([b_seqs, b_seqs, b_seqs], dtype=jnp.int32)

            ref_3d, ref_cache_u8 = _exact_decode_ref(
                np.asarray(q_j, dtype=np.float32),
                np.asarray(kv_merged),
                cache_u8,
                np.asarray(kv_list, dtype=np.int32),
                page_indices,
                cu_q,
                cu_kv,
                b_seqs,
                sm_scale=0.125,
                sliding_window=sliding_window,
                dq=dq,
            )
            ref_pk2 = np.concatenate(
                [
                    ref_3d[:, :8, :64].reshape(b_tok, 512),
                    ref_3d[:, 8:, 64:].reshape(b_tok, 512),
                ],
                axis=-1,
            )
            heads_f = ref_pk2.astype(np.float32).reshape(b_tok, 16, 64)
            inv_r = 1.0 / np.sqrt(np.mean(heads_f * heads_f, axis=-1, keepdims=True) + 1e-5)
            normed = jnp.asarray(heads_f * inv_r, dtype=jnp.bfloat16).reshape(b_tok, 1024)
            ref_gated = np.asarray(
                jax.nn.sigmoid(gate_j.astype(jnp.float32)).astype(jnp.bfloat16) * normed,
                dtype=np.float32,
            )

            out_g, upd_cache = run_decode_fast_rpa(
                q_5d,
                kv_merged,
                None,
                _fp8_from_u8(cache_u8),
                kv_lens_j,
                jnp.asarray(page_indices),
                jnp.asarray(cu_q),
                jnp.asarray(cu_kv),
                dist_j,
                sm_scale=0.125,
                sliding_window=sliding_window,
                k_scale=1.0,
                v_scale=1.0,
                mode_str="v2",
                decode_query_size=dq,
                fused_out_pack=2,
                prepared_inputs=True,
                gate=gate_j,
                out_norm_eps=1e-5,
                interpret=True,
            )
            out_np = np.asarray(out_g, dtype=np.float32)
            self.assertFalse(np.isnan(out_np).any(), f"NaN in output for sw={sliding_window}, dq={dq}")
            active_rows = b_seqs * dq
            cos = _cosine(out_np[:active_rows], ref_gated[:active_rows])
            self.assertGreater(cos, 0.99998, f"Low cosine {cos} for sw={sliding_window}, dq={dq}")
            upd_u8 = np.asarray(jax.lax.bitcast_convert_type(upd_cache, jnp.uint8))
            np.testing.assert_array_equal(upd_u8, ref_cache_u8)

    def test_finer_prefill_buckets_and_tpu7x_mosaic_lowering(self):
        from jax import export as jax_export
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
        from sglang_tpu.kernels.twoshot_all_reduce import bridge_twoshot_all_reduce
        from sglang_tpu.moe.pallas.fused_rs_muse.decode_norm_router import (
            residual_and_next_norm_pallas_jax,
            rmsnorm_pallas_jax,
        )
        from sglang_tpu.muse_prefill_kernels.attn_kernels import entry_exit, kernel_a3
        from sglang_tpu.muse_prefill_kernels.integration import (
            _a1_first_rows,
            _a1_part_rows,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_FINER_PREFILL_BUCKETS": "1"}):
            for t_tot in (3072, 4096, 5120, 6144, 7168, 8192):
                t_sh = t_tot // 8
                r_w = _a1_part_rows(t_sh, (2, 3, 2, 1), n_default_parts=4)
                self.assertEqual(sum(r_w), t_sh)
                self.assertTrue(all(x > 0 and x % 128 == 0 for x in r_w))
                r0 = _a1_first_rows(t_sh, "v5x4", ())
                self.assertEqual(r0 % 128, 0)

        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_RPA_V3_DECODE_FAST_V2": "1",
                    "SGLANG_TPU_MUSE_KERNEL_DECODE_NORM_FAST": "1",
                    "SGLANG_TPU_MUSE_KERNEL_A3_ZERO_SPILL_PROLOGUE": "1",
                },
            ),
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            # 1. Lower D3v2 SWA stream0b + pk2 + onorm + gate (dq=1 & dq=4) & Full pps=129 streamdyn
            for sw, pps, b_seqs, dq, m_str in [
                (2049, 129, 64, 1, "1"),
                (None, 129, 64, 1, "streamdyn"),
                (2049, 129, 64, 4, "1"),
            ]:
                b_tok = b_seqs * dq
                hlo = jax_export.export(
                    jax.jit(
                        lambda q, kv, g, c, kl, pi, cq, ckv, d, _m=m_str: run_decode_fast_rpa(
                            q,
                            kv,
                            None,
                            c,
                            kl,
                            pi,
                            cq,
                            ckv,
                            d,
                            sm_scale=0.125,
                            sliding_window=sw,
                            k_scale=1.0,
                            v_scale=1.0,
                            mode_str=_m,
                            decode_query_size=dq,
                            fused_out_pack=2,
                            prepared_inputs=True,
                            gate=g,
                            out_norm_eps=1e-5,
                        )
                    ),
                    platforms=["tpu"],
                )(
                    jax.ShapeDtypeStruct((1, b_tok, 8, 2, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((b_tok, 1, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((b_tok, 1024), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1024, 128, 1, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((b_seqs,), jnp.int32),
                    jax.ShapeDtypeStruct((b_seqs * pps,), jnp.int32),
                    jax.ShapeDtypeStruct((b_seqs + 1,), jnp.int32),
                    jax.ShapeDtypeStruct((b_seqs + 1,), jnp.int32),
                    jax.ShapeDtypeStruct((3,), jnp.int32),
                ).mlir_module()
                self.assertIn("RPAd_fast", hlo)

            # 1b. Lower full ragged_paged_attention entry point with _prepared_inputs=True + _decode_gate_in_values=True
            # Default _decode_fast="v2" accelerates dq=1 SWA while leaving dq>1 to RPAd-zt (VERIFY_RPA_FAST=1);
            # explicit _decode_fast="v2_multiq" lowers multi-Q RPAd_fast.
            for dq_test, df_mode, expect_fast in [
                (1, "v2", True),
                (2, "v2", False),
                (2, "v2_multiq", True),
            ]:
                b_t = 64 * dq_test
                hlo_rpa_entry = jax_export.export(
                    jax.jit(
                        lambda q, kv, g, c, kl, pi, cq, ckv, d, _dq=dq_test, _df=df_mode: ragged_paged_attention(
                            q,
                            kv,
                            g,
                            c,
                            kl,
                            pi,
                            cq,
                            ckv,
                            d,
                            None,
                            None,
                            sm_scale=0.125,
                            sliding_window=2049,
                            k_scale=1.0,
                            v_scale=1.0,
                            decode_query_size=_dq,
                            _fused_out_pack=2,
                            _prepared_inputs=True,
                            _skip_mixed=True,
                            _decode_fast=_df,
                            _decode_out_norm_eps=1e-5,
                            _decode_gate_in_values=True,
                        )
                    ),
                    platforms=["tpu"],
                )(
                    jax.ShapeDtypeStruct((1, b_t, 8, 2, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((b_t, 1, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((b_t, 1024), jnp.bfloat16),
                    jax.ShapeDtypeStruct((1024, 128, 1, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((64,), jnp.int32),
                    jax.ShapeDtypeStruct((64 * 129,), jnp.int32),
                    jax.ShapeDtypeStruct((65,), jnp.int32),
                    jax.ShapeDtypeStruct((65,), jnp.int32),
                    jax.ShapeDtypeStruct((3,), jnp.int32),
                ).mlir_module()
                if expect_fast:
                    self.assertIn("RPAd_fast", hlo_rpa_entry)
                else:
                    self.assertNotIn("RPAd_fast", hlo_rpa_entry)

            # 2. Lower Decode Norm Fast
            hlo_norm = jax_export.export(
                jax.jit(
                    lambda s, b, g, a, bt, ng: (
                        rmsnorm_pallas_jax(
                            s, g, eps=1e-5, has_gain=True, scale_input=False, out_bf16=True
                        ),
                        residual_and_next_norm_pallas_jax(
                            s,
                            b,
                            g,
                            a,
                            bt,
                            ng,
                            branch_eps=1e-5,
                            has_branch_gain=False,
                            branch_scale_input=False,
                            next_eps=1e-5,
                            next_out_bf16=False,
                        ),
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((64, 8192), jnp.float32),
                jax.ShapeDtypeStruct((64, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
            ).mlir_module()
            self.assertIn("_lf", hlo_norm)

            # 3. Lower Kernel A3 zero-spill prologue
            hlo_a3 = jax_export.export(
                jax.jit(
                    lambda a, g: kernel_a3.pallas_gate_static_fp8_quant(
                        a, g, tm=1024, head_norm_eps=1e-5, rpa_packed_out=True
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((8192, 16, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((8192, 1024), jnp.bfloat16),
            ).mlir_module()
            self.assertIn("pallas_gate_static_fp8_quant", hlo_a3)

            # 4. Lower bridge_twoshot_all_reduce & sharded_rope_gather_128 (0 partition_id!)
            devs = jax.devices()[:8]
            if len(devs) == 8:
                mesh = Mesh(np.array(devs), ("model",))
                sh_in = jax.NamedSharding(mesh, P("model"))
                sh_rep = jax.NamedSharding(mesh, P())
                hlo_br = jax_export.export(
                    jax.jit(
                        jax.shard_map(
                            lambda x: bridge_twoshot_all_reduce(x, world_size=8),
                            mesh=mesh,
                            in_specs=(P("model"),),
                            out_specs=P(),
                            check_vma=False,
                        )
                    ),
                    platforms=["tpu"],
                )(
                    jax.ShapeDtypeStruct((512, 8192), jnp.bfloat16, sharding=sh_in)
                ).mlir_module()
                self.assertIn("bridge_twoshot_all_reduce_tp8", hlo_br)

                hlo_rope = jax_export.export(
                    jax.jit(
                        jax.shard_map(
                            lambda pos, tbl: entry_exit.sharded_rope_gather_128(
                                pos, tbl, tp_axis="model", tp_size=8, pos_pre_sliced=True
                            ),
                            mesh=mesh,
                            in_specs=(P("model"), P()),
                            out_specs=(P("model"), P("model")),
                            check_vma=False,
                        )
                    ),
                    platforms=["tpu"],
                )(
                    jax.ShapeDtypeStruct((8192,), jnp.int32, sharding=sh_in),
                    jax.ShapeDtypeStruct((16384, 128), jnp.float32, sharding=sh_rep),
                ).mlir_module()
                self.assertNotIn("partition_id", hlo_rope)

    def test_decode_attn_gate_fuse_wiring_and_fallback(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        import torch
        from torch import nn
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang_tpu.models.muse_spark import MuseSparkAttention

        attn = MuseSparkAttention.__new__(MuseSparkAttention)
        nn.Module.__init__(attn)
        attn.head_dim = 128
        attn.num_heads = 2
        attn.num_kv_heads = 1
        attn.q_size = 256
        attn.kv_size = 128
        attn.norm_eps = 1e-5
        attn.decode_qkv_epilogue = True
        attn.decode_attn_gate_fuse = True
        attn.query_scale = 1.0
        attn.rotary_emb = None
        attn._qkvg_w8k = None
        attn._qkvg_weight = None
        attn._o_proj_w8k = None
        attn.qkv_proj = Mock(side_effect=lambda x: (torch.zeros(x.shape[0], 512, dtype=x.dtype), None))
        attn.gate_proj = Mock(side_effect=lambda x: (torch.ones(x.shape[0], 256, dtype=x.dtype), None))
        attn.o_proj = Mock(side_effect=lambda x, skip_all_reduce=False: (x, None))
        attn.attn = Mock()
        attn._per_head_normalize = Mock(side_effect=lambda x: x)

        x = torch.ones((16, 512), dtype=torch.bfloat16)
        pos = torch.arange(16, dtype=torch.int32)
        fb = SimpleNamespace(forward_mode=ForwardMode.DECODE, attn_backend=None)

        fused_calls = []

        def _fake_fused(*args, gate=None, return_post_fused=False, **kw):
            fused_calls.append((gate is not None, return_post_fused))
            out = torch.full((16, 256), 3.0, dtype=torch.bfloat16)
            return (out, gate is not None) if return_post_fused else out

        with (
            patch("sglang_tpu.models.muse_spark.can_use_muse_decode_qkv_epilogue", return_value=True),
            patch("sglang_tpu.models.muse_spark.muse_decode_fused_qkv_attn", side_effect=_fake_fused),
        ):
            got = attn(pos, x, fb)
            self.assertEqual(fused_calls, [(True, True)])
            attn._per_head_normalize.assert_not_called()
            self.assertEqual(attn.gate_proj.call_count, 1)
            self.assertTrue(torch.equal(got, torch.full((16, 256), 3.0, dtype=torch.bfloat16)))

            # return_attn_and_gate=True bypasses gate fusion cleanly
            fused_calls.clear()
            attn._per_head_normalize.reset_mock()
            a_raw, g_raw = attn(pos, x, fb, return_attn_and_gate=True)
            self.assertEqual(fused_calls, [(False, False)])
            attn._per_head_normalize.assert_not_called()
            self.assertEqual(tuple(g_raw.shape), (16, 256))

        # When post_fused=False fallback occurs, gate_proj is NOT called twice
        attn.gate_proj.reset_mock()
        attn._per_head_normalize.reset_mock()
        with (
            patch("sglang_tpu.models.muse_spark.can_use_muse_decode_qkv_epilogue", return_value=True),
            patch(
                "sglang_tpu.models.muse_spark.muse_decode_fused_qkv_attn",
                side_effect=lambda *a, **kw: (torch.ones(16, 256, dtype=torch.bfloat16), False),
            ),
        ):
            _ = attn(pos, x, fb)
            self.assertEqual(attn.gate_proj.call_count, 1)
            self.assertEqual(attn._per_head_normalize.call_count, 1)

    def test_integration_env_bool_guards_avoid_envvar_bool_error(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from sglang_tpu.muse_prefill_kernels import integration as mp_int
        from sglang_tpu.envs import envs as tpu_envs

        # Must never trigger EnvVar.__bool__() RuntimeError
        for name in (
            "SGLANG_TPU_MUSE_KERNEL_FINER_PREFILL_BUCKETS",
            "SGLANG_TPU_MUSE_KERNEL_A3_ZERO_SPILL_PROLOGUE",
            "SGLANG_TPU_MUSE_KERNEL_EP_TP2_PREFILL",
            "SGLANG_TPU_MUSE_KERNEL_NONEXISTENT_FLAG",
        ):
            self.assertFalse(mp_int._env_bool(name))

        with tpu_envs.SGLANG_TPU_MUSE_KERNEL_FINER_PREFILL_BUCKETS.override(True):
            self.assertTrue(mp_int._env_bool("SGLANG_TPU_MUSE_KERNEL_FINER_PREFILL_BUCKETS"))

        # Exercise MusePrefillKernels.from_model past line 1664 without RuntimeError
        stub_cfg = SimpleNamespace(
            num_attention_heads=128,
            num_key_value_heads=16,
            qk_scale_factor=1.0,
            output_multiplier=0.1,
            routed_scaling_factor=1.0,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            hidden_size=8192,
            moe_hidden_size=4096,
            num_local_experts=256,
            num_experts_per_tok=8,
            head_dim=64,
        )
        stub_model = SimpleNamespace(
            model=SimpleNamespace(pp_group=SimpleNamespace(world_size=1)),
            text_config=stub_cfg,
        )
        stub_par = SimpleNamespace(tp_size=8, attn_tp_size=8, moe_ep_size=8, moe_tp_size=1, attn_dcp_size=1)
        with (
            patch("sglang.srt.runtime_context.get_parallel", return_value=stub_par),
            patch("sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=False),
        ):
            self.assertIsNone(mp_int.MusePrefillKernels.from_model(stub_model))

    def test_wave6_prefill_rpa_sw2048_full_fuse_a1_and_d_t128_mosaic_export(self):
        from jax import export as jax_export
        from jax.sharding import Mesh
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )
        from sglang_tpu.envs import envs
        from sglang_tpu.muse_prefill_kernels.integration import _a1_body
        from sglang_tpu.muse_prefill_kernels.moe_bridge_kernels import kernel_d
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        self.assertFalse(envs.SGLANG_TPU_RPA_V3_PREFILL_SW2048_FUSE.get())
        self.assertEqual(envs.SGLANG_TPU_MUSE_KERNEL_A1_FAST_CHUNKS.get(), "")
        self.assertFalse(envs.SGLANG_TPU_MUSE_KERNEL_D_T128_FAST.get())

        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_RPA_V3_PREFILL_SW2048_FUSE": "1",
                    "SGLANG_TPU_MUSE_KERNEL_D_T128_FAST": "1",
                    "SGLANG_TPU_MUSE_KERNEL_D_SALU_OVERLAP": "1",
                },
            ),
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            # 1. Export RPAm sw=2048 + sw2048_fuse (a3_out_layout=True, m0_ring_fuse)
            opts_sw2048 = frozenset(
                {
                    "stationary_q_m0_ring_fuse",
                    "skip_empty_cdma",
                    "mixed_s3",
                    "sw2048_fuse",
                }
            )
            hlo_rpam = jax_export.export(
                jax.jit(
                    lambda q, kv, c, kl, pi, cq, ckv, d: ragged_paged_attention(
                        q,
                        kv,
                        None,
                        c,
                        kl,
                        pi,
                        cq,
                        ckv,
                        d,
                        None,
                        None,
                        sm_scale=0.125,
                        sliding_window=2048,
                        k_scale=1.0,
                        v_scale=1.0,
                        m_block_sizes=(256, 2048, 256, 256),
                        _prefill_opts=opts_sw2048,
                        _fused_out_pack=2,
                        _prepared_inputs=True,
                        _skip_decode=True,
                        _a3_out_layout=True,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, 8192, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((8192, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((512, 128, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8,), jnp.int32),
                jax.ShapeDtypeStruct((8 * 129,), jnp.int32),
                jax.ShapeDtypeStruct((9,), jnp.int32),
                jax.ShapeDtypeStruct((9,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("RPAm", hlo_rpam)
            self.assertIn("sw2048_fuse", hlo_rpam)

            # 2. Export Kernel D t128 + c16h + salu_overlap
            hlo_d128 = jax_export.export(
                jax.jit(
                    lambda ob, ip, tw, sm, wu, wus, pfw, af, bf, niw: kernel_d.pallas_combine_unpool_residual_nextnorm_4slab(
                        ob, ip, tw, sm, wu, wus, pfw, af, bf, niw
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((8192, 32, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((1024, 8), jnp.int32),
                jax.ShapeDtypeStruct((1024, 8), jnp.float32),
                jax.ShapeDtypeStruct((1024, 8192), jnp.float32),
                jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
                jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
                jax.ShapeDtypeStruct((8192,), jnp.float32),
                jax.ShapeDtypeStruct((8192,), jnp.bfloat16),
            ).mlir_module()
            self.assertIn("pallas_d_combine_4slab_v3_t128", hlo_d128)
            self.assertIn("_c16so", hlo_d128)

            # 3. Export A1 v6_s8str128_bf16kv with A1_FAST_CHUNKS="512" & "1024" -> tm256!
            devs = jax.devices()[:8]
            if len(devs) == 8:
                mesh = Mesh(np.array(devs), ("model",))
                sh_m = jax.NamedSharding(mesh, P("model"))
                sh_m0 = jax.NamedSharding(mesh, P("model", None))
                sh_r = jax.NamedSharding(mesh, P())
                for fch, exp_sub in [
                    ("512", "off512_h512_tm256"),
                    ("1024", "off0_h1024_tm256"),
                ]:
                    with patch.dict(
                        os.environ, {"SGLANG_TPU_MUSE_KERNEL_A1_FAST_CHUNKS": fch}
                    ):
                        body = _a1_body(16, 2, "v6_s8str128_bf16kv", rpa_layout=True)
                        hlo_a1 = jax_export.export(
                            jax.jit(
                                jax.shard_map(
                                    body,
                                    mesh=mesh,
                                    in_specs=(
                                        P("model", None),
                                        P("model"),
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
                            ),
                            platforms=["tpu"],
                        )(
                            jax.ShapeDtypeStruct((8192, 8192), jnp.bfloat16, sharding=sh_m0),
                            jax.ShapeDtypeStruct((8 * 8192, 2048), jnp.float8_e4m3fn, sharding=sh_m),
                            jax.ShapeDtypeStruct((8, 2048), jnp.float32, sharding=sh_m),
                            jax.ShapeDtypeStruct((8 * 8192, 256), jnp.float8_e4m3fn, sharding=sh_r),
                            jax.ShapeDtypeStruct((8, 256), jnp.float32, sharding=sh_r),
                            jax.ShapeDtypeStruct((1, 8192), jnp.bfloat16, sharding=sh_r),
                            jax.ShapeDtypeStruct((1, 8192), jnp.bfloat16, sharding=sh_r),
                            jax.ShapeDtypeStruct((8192, 128), jnp.float32, sharding=sh_r),
                            jax.ShapeDtypeStruct((8192, 128), jnp.float32, sharding=sh_r),
                        ).mlir_module()
                        self.assertIn(exp_sub, hlo_a1)
                        self.assertNotIn("partition_id", hlo_a1)

    def test_v6_s8str128_outlier1000_scale_no_saturation(self):
        from jax import export as jax_export
        from sglang_tpu.muse_prefill_kernels.attn_kernels import kernel_a1
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        self.assertEqual(kernel_a1.S8_SCALE_PACK_MUL, 8.0)
        self.assertEqual(kernel_a1.S8_SCALE_UNPACK_MUL, 0.125)
        max_repr_amax = kernel_a1.FP8_MAX * 127.0 / kernel_a1.S8_SCALE_PACK_MUL
        self.assertGreater(max_repr_amax, 7000.0)

        rng = np.random.default_rng(890)
        rows, k_dim = 64, 8192
        gain_np = np.clip(1.2 + 0.2 * rng.standard_normal((1, k_dim)), 0.8, 3.0).astype(
            np.float32
        )
        inv_gain_np = (1.0 / gain_np).astype(np.float32)
        x_np = (rng.standard_normal((rows, k_dim)) * 0.4).astype(np.float32)
        # Inject real long-8k outlier channels where |x / gain| = 1000.0 and 3500.0
        # (well above the old 111.125 clip cliff!).
        x_np[:, 0::512] = 1000.0 * gain_np[:, 0::512]
        x_np[::2, 128::1024] = -3500.0 * gain_np[:, 128::1024]

        x_bf16 = jnp.asarray(x_np, dtype=jnp.bfloat16)
        gain_bf16 = jnp.asarray(gain_np, dtype=jnp.bfloat16)
        inv_gain_bf16 = jnp.asarray(inv_gain_np, dtype=jnp.bfloat16)
        wqg_np = (rng.standard_normal((k_dim, 512)) * 0.03).astype(np.float32)
        wqg_f8 = jnp.asarray(wqg_np, dtype=jnp.float8_e4m3fn)

        ref_dot = np.asarray(
            jnp.dot(
                x_bf16.astype(jnp.float32),
                wqg_f8.astype(jnp.float32),
            ),
            dtype=np.float32,
        )

        for per_group in (True, False):
            packed = kernel_a1.quant_chunk_s8_packed(
                x_bf16, inv_gain_bf16, per_group_128=per_group
            )
            self.assertEqual(packed.shape, (rows, k_dim + 128))
            s_pack_f8 = jax.lax.bitcast_convert_type(
                packed[:, k_dim:], jnp.float8_e4m3fn
            )
            if per_group:
                sc_64 = (
                    (
                        s_pack_f8[:, :64].astype(jnp.float32)
                        + s_pack_f8[:, 64:].astype(jnp.float32) * jnp.float32(1.0 / 16.0)
                    )
                    * jnp.float32(kernel_a1.S8_SCALE_UNPACK_MUL)
                ).astype(jnp.bfloat16)
                deq = (
                    packed[:, :k_dim].reshape(rows, 64, 128).astype(jnp.bfloat16)
                    * sc_64[:, :, None]
                ).reshape(rows, k_dim) * gain_bf16
            else:
                sc_row = (
                    (
                        s_pack_f8[:, 0:1].astype(jnp.float32)
                        + s_pack_f8[:, 64:65].astype(jnp.float32)
                        * jnp.float32(1.0 / 16.0)
                    )
                    * jnp.float32(kernel_a1.S8_SCALE_UNPACK_MUL)
                ).astype(jnp.bfloat16)
                deq = (packed[:, :k_dim].astype(jnp.bfloat16) * sc_row) * gain_bf16

            got_dot = np.asarray(
                jnp.dot(deq.astype(jnp.float32), wqg_f8.astype(jnp.float32)),
                dtype=np.float32,
            )
            cos = _cosine(got_dot, ref_dot)
            self.assertGreater(
                cos,
                0.9995 if per_group else 0.995,
                f"Outlier1000/3500 scale clipped for per_group={per_group}: cos={cos}",
            )

        with (
            patch(
                "sglang_tpu.pallas_export.get_pallas_tpu_device_kind",
                return_value="TPU7x",
            ),
            patch(
                "sglang_tpu.pallas_export.get_tpu_generation",
                return_value=7,
            ),
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            hlo_q0 = jax_export.export(
                jax.jit(
                    lambda x, ig: kernel_a1.pallas_quant_chunk0_s8_unsliced(
                        x, ig, c0_rows=512, strided_128=True
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1024, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((1, 8192), jnp.bfloat16),
            ).mlir_module()
            self.assertIn("pallas_quant_c0_s8str128_h512_tm256", hlo_q0)

        import inspect

        for fn_obj in (
            kernel_a1.quant_chunk_s8_packed,
            kernel_a1.pallas_a1_qg_1way_fkv,
            kernel_a1.pallas_quant_chunk0_s8_unsliced,
            kernel_a1.pallas_sp_kv_local_bf16_fkvq,
            kernel_a1.pallas_a1_qg_1way_fkv_zc,
        ):
            src_fn = inspect.getsource(fn_obj)
            self.assertNotIn("512.0", src_fn, f"Stale 512.0 scale in {fn_obj.__name__}")

    def test_decode_gated_custom_op_passes_non_none_gate_values(self):
        import dataclasses
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        import sglang_tpu.attention_pallas_rpa_base as rpa_base
        from sglang_tpu.attention_pallas_rpa_v3 import (
            RPA_V3_SPEC,
            TpuPallasRpaV3AttnBackend,
            _decode_gated_spec,
            _fused_out_spec,
            _prepared_spec,
            _skip_mixed_spec,
        )

        prep_plain = _prepared_spec(_skip_mixed_spec(_fused_out_spec(2)))
        prep_gated = _decode_gated_spec(prep_plain, 1e-5)
        self.assertTrue(prep_plain.prepared_inputs)
        self.assertFalse(prep_plain.keep_values_when_prepared)
        self.assertTrue(prep_gated.prepared_inputs)
        self.assertTrue(prep_gated.keep_values_when_prepared)
        self.assertNotEqual(prep_plain.op_name, prep_gated.op_name)

        # Verify _build_pallas_kernel_op's custom_op body forwards `values` (`gate`)
        # when keep_values_when_prepared=True, while still passing None when False!
        captured_values = []
        orig_custom_op = torch.library.custom_op

        def _cpu_custom_op(name, fn=None, /, *, mutates_args=(), device_types=None, schema=None):
            return orig_custom_op(
                name,
                fn,
                mutates_args=mutates_args,
                device_types=["cpu"],
                schema=schema,
            )

        def _fake_custom_jax_kernel(wrapped_fn, **_kw):
            def _runner(*args):
                q_t, _k_t, v_t, kv_c = args[:4]
                captured_values.append(v_t)
                b_tok = q_t.shape[1]
                base_normed = torch.full((b_tok, 1024), 2.0, dtype=torch.bfloat16)
                if v_t is not None:
                    sig_g = torch.sigmoid(v_t.to(torch.float32)).to(torch.bfloat16)
                    return sig_g * base_normed, kv_c
                return base_normed, kv_c

            return _runner

        test_plain = dataclasses.replace(
            prep_plain, op_name=f"{prep_plain.op_name}_unit_plain"
        )
        test_gated = dataclasses.replace(
            prep_gated, op_name=f"{prep_gated.op_name}_unit_gated"
        )
        with (
            patch("torch.library.custom_op", side_effect=_cpu_custom_op),
            patch(
                "torch_tpu._internal.pallas.custom_jax_kernel",
                side_effect=_fake_custom_jax_kernel,
            ),
        ):
            op_plain = rpa_base._build_pallas_kernel_op(test_plain)
            op_gated = rpa_base._build_pallas_kernel_op(test_gated)

        b_tok = 16
        q_5d = torch.ones((1, b_tok, 8, 2, 128), dtype=torch.bfloat16)
        kv_m = torch.zeros((b_tok, 1, 4, 128), dtype=torch.float8_e4m3fn)
        # Non-contiguous slice view from qkvg.split([1280, 1024], dim=-1) with gate = -10.0
        # so sigmoid(gate) ~ 4.5e-5 != 1.0!
        qkvg = torch.cat(
            [
                torch.zeros((b_tok, 1280), dtype=torch.bfloat16),
                torch.full((b_tok, 1024), -10.0, dtype=torch.bfloat16),
            ],
            dim=-1,
        )
        _, gate_view = qkvg.split([1280, 1024], dim=-1)
        self.assertFalse(gate_view.is_contiguous())

        kv_cache = torch.zeros((32, 128, 1, 4, 128), dtype=torch.float8_e4m3fn)
        kv_lens = torch.full((b_tok,), 128, dtype=torch.int32)
        page_idx = torch.arange(32, dtype=torch.int32)
        cu_q = torch.arange(b_tok + 1, dtype=torch.int32)
        cu_kv = torch.arange(b_tok + 1, dtype=torch.int32) * 128
        dist = torch.tensor([b_tok, b_tok, b_tok], dtype=torch.int32)

        out_plain = op_plain(
            q_5d, kv_m, gate_view, kv_cache, kv_lens, page_idx, cu_q, cu_kv, dist
        )
        self.assertIsNone(captured_values[-1])
        self.assertTrue(torch.equal(out_plain, torch.full((b_tok, 1024), 2.0, dtype=torch.bfloat16)))

        # Test through TpuPallasRpaV3AttnBackend.forward_decode_prepared_gated too
        backend = TpuPallasRpaV3AttnBackend.__new__(TpuPallasRpaV3AttnBackend)
        backend._decode_prepared_op = op_plain
        backend._decode_prepared_gated_op = op_gated
        backend._dcp = 1
        backend.kv_cache_quant_dtype = torch.float8_e4m3fn
        backend.token_to_kv_pool = SimpleNamespace()
        backend.forward_metadata = SimpleNamespace(
            decode_query_size=1,
            cp_write_segments=None,
            cu_q_lens=cu_q,
            cu_kv_lens=cu_kv,
            active_num_tokens=torch.tensor(b_tok, dtype=torch.int32),
        )
        backend._layer_page_indices = lambda lid, is_swa_layer: page_idx
        backend._kv_lens = lambda fb: kv_lens
        backend._layer_distribution = lambda is_swa_layer: dist
        layer = SimpleNamespace(
            layer_id=0,
            is_cross_attention=False,
            tp_q_head_num=16,
            tp_k_head_num=2,
            qk_head_dim=64,
            v_head_dim=64,
            logit_cap=0,
            sliding_window_size=2047,
            k_scale_float=1.0,
            v_scale_float=1.0,
            scaling=0.125,
        )
        fb = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: True, is_target_verify=lambda: False
            ),
            capture_hidden_mode=None,
        )
        with patch(
            "sglang_tpu.attention_pallas_rpa_v3._fused_kv_operands",
            return_value=(kv_cache, None),
        ):
            out_gated = backend.forward_decode_prepared_gated(
                q_5d, kv_m, gate_view, layer, fb
            )
        self.assertIsNotNone(captured_values[-1])
        self.assertTrue(captured_values[-1].is_contiguous())
        expected_gated = (
            torch.sigmoid(gate_view.to(torch.float32)).to(torch.bfloat16) * 2.0
        )
        self.assertTrue(torch.equal(out_gated, expected_gated))
        self.assertLess(float(out_gated.abs().max()), 1e-3)

    def test_decode_qkv_epilogue_forward_context_hybrid_and_rope_hoist(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from torch import nn
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang_tpu.decode_qkv_epilogue_tpu import (
            _resolve_decode_attn_backend,
            can_use_muse_decode_qkv_epilogue,
            can_use_muse_decode_rope_step,
            muse_decode_fused_qkv_attn,
        )
        from sglang_tpu.models.muse_spark import (
            MuseSparkAttention,
            MuseSparkDecoderLayer,
            MuseSparkTextModel,
        )

        # 1. Build a real HybridAttnBackend (prefill lacks prepared decode, decode supports it)
        prefill_be = SimpleNamespace(needs_cpu_seq_lens=False)
        gated_calls = []
        plain_calls = []
        decode_be = SimpleNamespace(
            needs_cpu_seq_lens=False,
            can_forward_decode_prepared=lambda layer, fb: (
                fb.forward_mode.is_decode() or fb.forward_mode.is_target_verify()
            ),
            forward_decode_prepared=lambda q5, kvm, layer, fb: (
                plain_calls.append((tuple(q5.shape), tuple(kvm.shape), layer.layer_id))
                or torch.full((q5.shape[1], 1024), 1.5, dtype=torch.bfloat16)
            ),
            forward_decode_prepared_gated=lambda q5, kvm, g, layer, fb: (
                gated_calls.append(
                    (tuple(q5.shape), tuple(kvm.shape), tuple(g.shape), layer.layer_id)
                )
                or (
                    torch.sigmoid(g.to(torch.float32)).to(torch.bfloat16)
                    * torch.full((q5.shape[1], 1024), 1.5, dtype=torch.bfloat16)
                )
            ),
        )
        hybrid = HybridAttnBackend.__new__(HybridAttnBackend)
        hybrid.prefill_backend = prefill_be
        hybrid.decode_backend = decode_be
        hybrid.spec_attn_is_decode = True
        hybrid.spec_attn_is_prefill = False

        # forward_batch intentionally has NO .attn_backend attribute (live serving contract!)
        fb_decode = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        fb_extend = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        fb_verify = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        fb_draft_ext = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
        self.assertFalse(hasattr(fb_decode, "attn_backend"))

        rope_obj = SimpleNamespace(cos_sin_pairs=torch.ones((64, 128), dtype=torch.float32))
        swa_layer = SimpleNamespace(
            layer_id=0, k_scale_float=1.0, v_scale_float=1.0, sliding_window_size=2047
        )
        nope_layer = SimpleNamespace(
            layer_id=3, k_scale_float=1.0, v_scale_float=1.0, sliding_window_size=-1
        )
        qkv = torch.ones((16, 1280), dtype=torch.bfloat16)
        gate = torch.full((16, 1024), -4.0, dtype=torch.bfloat16)
        pos = torch.arange(16, dtype=torch.int64)

        rope_step_calls = []
        epi_rope_calls = []
        epi_nope_calls = []

        def _fake_rope_step(p_i32, pairs):
            rope_step_calls.append(tuple(p_i32.shape))
            return pairs.new_ones((32, 2, 128))

        def _fake_epi_rope(qkv_in, cs_in, *_a):
            epi_rope_calls.append((tuple(qkv_in.shape), cs_in))
            b = qkv_in.shape[0]
            return (
                qkv_in.new_ones((1, b, 8, 2, 128)),
                torch.zeros((b, 1, 4, 128), dtype=torch.float8_e4m3fn),
            )

        def _fake_epi_nope(qkv_in, *_a):
            epi_nope_calls.append(tuple(qkv_in.shape))
            b = qkv_in.shape[0]
            return (
                qkv_in.new_ones((1, b, 8, 2, 128)),
                torch.zeros((b, 1, 4, 128), dtype=torch.float8_e4m3fn),
            )

        with forward_context(ForwardContext(attn_backend=hybrid)):
            self.assertIs(_resolve_decode_attn_backend(fb_decode), decode_be)
            self.assertIs(_resolve_decode_attn_backend(fb_extend), prefill_be)
            self.assertTrue(
                can_use_muse_decode_qkv_epilogue(qkv, rope_obj, swa_layer, fb_decode)
            )
            self.assertTrue(
                can_use_muse_decode_qkv_epilogue(qkv, None, nope_layer, fb_decode)
            )
            self.assertFalse(
                can_use_muse_decode_qkv_epilogue(qkv, rope_obj, swa_layer, fb_extend)
            )
            self.assertFalse(
                can_use_muse_decode_qkv_epilogue(qkv, rope_obj, swa_layer, fb_draft_ext)
            )
            self.assertFalse(
                can_use_muse_decode_rope_step(16, rope_obj, swa_layer, fb_verify)
            )
            with patch.dict(os.environ, {"SGLANG_TPU_MUSE_KERNEL_VERIFY_RPA_FAST": "1"}):
                self.assertTrue(
                    can_use_muse_decode_rope_step(16, rope_obj, swa_layer, fb_verify)
                )

            # 2. Verify zero-graph-break Dynamo fullgraph=True tracing through ForwardContext + HybridAttnBackend
            @torch.compile(backend="eager", fullgraph=True)
            def _traced_check(x_qkv):
                ok_rope = can_use_muse_decode_qkv_epilogue(
                    x_qkv, rope_obj, swa_layer, fb_decode
                )
                ok_nope = can_use_muse_decode_qkv_epilogue(
                    x_qkv, None, nope_layer, fb_decode
                )
                ok_step = can_use_muse_decode_rope_step(
                    x_qkv.shape[0], rope_obj, swa_layer, fb_decode
                )
                return x_qkv + (1.0 if (ok_rope and ok_nope and ok_step) else 0.0)

            traced_out = _traced_check(qkv)
            self.assertTrue(torch.equal(traced_out, qkv + 1.0))

            # 3. Verify MuseSparkTextModel.forward hoists get_decode_rope_step_op() to 1 call across RoPE+NoPE layers
            def _make_layer(lid: int, has_rope: bool):
                sa = MuseSparkAttention.__new__(MuseSparkAttention)
                nn.Module.__init__(sa)
                sa.head_dim = 64
                sa.num_heads = 16
                sa.num_kv_heads = 2
                sa.q_size = 1024
                sa.kv_size = 128
                sa.norm_eps = 1e-5
                sa.decode_qkv_epilogue = True
                sa.decode_attn_gate_fuse = True
                sa.query_scale = 1.0
                sa.rotary_emb = rope_obj if has_rope else None
                sa._qkvg_w8k = None
                sa._qkvg_weight = None
                sa._o_proj_w8k = None
                sa.qkv_proj = lambda h: (torch.ones((h.shape[0], 1280), dtype=h.dtype), None)
                sa.gate_proj = lambda h: (torch.full((h.shape[0], 1024), -2.0, dtype=h.dtype), None)
                sa.o_proj = SimpleNamespace(
                    tp_size=1,
                    reduce_results=False,
                    __call__=lambda g, skip_all_reduce=False: (
                        torch.zeros((g.shape[0], 64), dtype=g.dtype),
                        None,
                    ),
                )
                sa.o_proj = lambda g, skip_all_reduce=False: (
                    torch.zeros((g.shape[0], 64), dtype=g.dtype),
                    None,
                )
                sa.o_proj.tp_size = 1
                sa.attn = swa_layer if has_rope else nope_layer

                dl = MuseSparkDecoderLayer.__new__(MuseSparkDecoderLayer)
                nn.Module.__init__(dl)
                dl._decode_norm_layout = 0
                dl._decode_fused_o_proj_ar = 0
                dl._decode_fuse_target_verify = False
                dl.self_attn = sa
                dl.input_layernorm = lambda s: s.to(torch.bfloat16)
                dl.post_attention_layernorm = lambda a: a
                dl.post_attention_residual_gate = lambda s, a: s + a.float()
                dl.pre_feedforward_layernorm = lambda s: s.to(torch.bfloat16)
                dl.mlp = lambda fh, ri, decode=False: fh
                dl.post_feedforward_layernorm = lambda f: f
                dl.post_feedforward_residual_gate = lambda s, f: s + f.float()
                return dl

            tm = MuseSparkTextModel.__new__(MuseSparkTextModel)
            nn.Module.__init__(tm)
            tm.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
            tm.prefill_kernels = None
            tm.layers_to_capture = []
            tm._capture_slots_by_layer = {}
            tm._decode_norm_layout = 0
            tm.rotary_emb = rope_obj
            tm.start_layer = 0
            tm.end_layer = 4
            tm.layers = nn.ModuleList(
                [
                    _make_layer(0, True),
                    _make_layer(1, True),
                    _make_layer(2, False),  # NoPE Full-attn layer!
                    _make_layer(3, True),
                ]
            )
            tm.norm = lambda x: x

            with (
                patch(
                    "sglang_tpu.models.muse_spark.get_decode_rope_step_op",
                    return_value=_fake_rope_step,
                ),
                patch(
                    "sglang_tpu.decode_qkv_epilogue_tpu.get_decode_rope_step_op",
                    return_value=_fake_rope_step,
                ),
                patch(
                    "sglang_tpu.decode_qkv_epilogue_tpu.get_decode_qkv_epilogue_op",
                    side_effect=lambda hr: _fake_epi_rope if hr else _fake_epi_nope,
                ),
            ):
                inp_emb = torch.ones((16, 64), dtype=torch.bfloat16)
                _ = tm(
                    input_ids=torch.zeros((16,), dtype=torch.int32),
                    positions=pos,
                    forward_batch=fb_decode,
                    input_embeds=inp_emb,
                )
                # Hoisted: 1 rope_step_op launch (not 3!), 3 epi_rope_op sharing same tensor, 1 epi_nope_op, 4 gated RPAd!
                self.assertEqual(len(rope_step_calls), 1)
                self.assertEqual(len(epi_rope_calls), 3)
                self.assertIs(epi_rope_calls[0][1], epi_rope_calls[1][1])
                self.assertIs(epi_rope_calls[1][1], epi_rope_calls[2][1])
                self.assertEqual(len(epi_nope_calls), 1)
                self.assertEqual(len(gated_calls), 4)
                self.assertEqual(len(plain_calls), 0)

                # DRAFT_EXTEND must NOT call get_decode_rope_step_op at all (0 orphaned ops!)
                rope_step_calls.clear()
                rope_callable = SimpleNamespace(
                    cos_sin_pairs=rope_obj.cos_sin_pairs,
                    __call__=lambda p, q, k: (q, k),
                )
                rope_fn = lambda p, q, k: (q, k)
                rope_fn.cos_sin_pairs = rope_obj.cos_sin_pairs
                tm.rotary_emb = rope_fn
                _make_layer_fallback = lambda *a, **kw: torch.zeros((16, 1024), dtype=torch.bfloat16)
                for lyr in tm.layers:
                    if lyr.self_attn.rotary_emb is not None:
                        lyr.self_attn.rotary_emb = rope_fn
                    lyr.self_attn._per_head_normalize = lambda t: t
                    lyr.self_attn.attn = _make_layer_fallback
                _ = tm(
                    input_ids=torch.zeros((16,), dtype=torch.int32),
                    positions=pos,
                    forward_batch=fb_draft_ext,
                    input_embeds=inp_emb,
                )
                self.assertEqual(len(rope_step_calls), 0)

    def test_cold_decode_qkv_epilogue_dynamo_export_no_lock_break(self):
        import torch
        import torch._dynamo
        from torch._dynamo.exc import Unsupported
        import sglang_tpu.decode_qkv_epilogue_tpu as epi_mod

        def _real_epi_graph(pos_t, pairs_t, qkv_t):
            cs = epi_mod.get_decode_rope_step_op()(pos_t.to(torch.int32), pairs_t)
            q_rope, kv_rope = epi_mod.get_decode_qkv_epilogue_op(True)(
                qkv_t, cs, 1e-5, 1.0, 1.0, 1.0
            )
            q_nope, kv_nope = epi_mod.get_decode_qkv_epilogue_op(False)(
                qkv_t, 1e-5, 1.0, 1.0, 1.0
            )
            return q_rope + q_nope, kv_rope, kv_nope

        pos_t = torch.arange(64, dtype=torch.int64)
        pairs_t = torch.ones((128, 128), dtype=torch.float32)
        qkv_t = torch.ones((64, 1280), dtype=torch.bfloat16)

        # 1. Cold state (_ROPE_STEP_OP = None) reproduces #1025 lock error if not pre-warmed
        torch._dynamo.reset()
        epi_mod._ROPE_STEP_OP = None
        epi_mod._EPI_ROPE_OP = None
        epi_mod._EPI_NOPE_OP = None
        with self.assertRaises(Unsupported):
            torch._dynamo.export(_real_epi_graph)(pos_t, pairs_t, qkv_t)

        # 2. warm_decode_qkv_epilogue_ops() (called by MuseSparkAttention.__init__) populates
        # all 3 ops before torch.compile starts -> fullgraph Dynamo export succeeds with 0 breaks!
        torch._dynamo.reset()
        epi_mod._ROPE_STEP_OP = None
        epi_mod._EPI_ROPE_OP = None
        epi_mod._EPI_NOPE_OP = None
        epi_mod.warm_decode_qkv_epilogue_ops()
        self.assertIsNotNone(epi_mod._ROPE_STEP_OP)
        self.assertIsNotNone(epi_mod._EPI_ROPE_OP)
        self.assertIsNotNone(epi_mod._EPI_NOPE_OP)

        gm_warm, guards = torch._dynamo.export(_real_epi_graph)(pos_t, pairs_t, qkv_t)
        self.assertIsNotNone(guards)
        warm_str = str(gm_warm.graph)
        self.assertIn("muse_decode_rope_step_tables", warm_str)
        self.assertIn("muse_decode_qkv_epilogue_rope", warm_str)
        self.assertIn("muse_decode_qkv_epilogue_nope", warm_str)


if __name__ == "__main__":
    unittest.main()
