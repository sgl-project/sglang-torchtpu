"""Long-Context Prefill RPA v3 (`lc-rpa-prefill`) unit, CPU `interpret=True`, and TPU7x Mosaic export tests."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[5]
_TOUCHED_FILES = (
    "sglang_tpu/envs.py",
    "sglang_tpu/attention_pallas_rpa_v3.py",
    "sglang_tpu/attention/pallas/ragged_paged_attention_v3.py",
)


def _u8_to_fp8(arr_u8: np.ndarray) -> jax.Array:
    return jax.lax.bitcast_convert_type(
        jnp.asarray(arr_u8, dtype=jnp.uint8), jnp.float8_e4m3fn
    )


def _fp8_to_u8(arr_f8: jax.Array) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(arr_f8, jnp.uint8))


def _rand_fp8_u8(
    rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.35
) -> np.ndarray:
    vals = (rng.standard_normal(shape) * scale).astype(np.float32)
    return _fp8_to_u8(jnp.asarray(vals, dtype=jnp.float8_e4m3fn))


def _rand_muse_q_3d(
    rng: np.random.Generator, n_tok: int, scale: float = 0.3
) -> np.ndarray:
    """Muse head-packed `q_3d [T, 16, 128]`: heads 0..7 in `[:64]`, 8..15 in `[64:]`."""
    q = (rng.standard_normal((n_tok, 16, 128)) * scale).astype(np.float32)
    q[:, :8, 64:] = 0.0
    q[:, 8:, :64] = 0.0
    return q


def _pack_q_5d(q_3d: np.ndarray) -> jax.Array:
    t = q_3d.shape[0]
    return jnp.asarray(q_3d, dtype=jnp.bfloat16).reshape(1, t, 8, 2, 128)


def _pack_stock_cache_to_dense2_u8(stock_u8: np.ndarray) -> np.ndarray:
    n_pg, pg_sz = stock_u8.shape[0], stock_u8.shape[1]
    d2 = np.full((n_pg, pg_sz // 2, 1, 4, 128), 0x7F, dtype=np.uint8)
    d2[:, :, 0, 0, :] = stock_u8[:, 0::2, 0, 0, :]
    d2[:, :, 0, 1, :] = stock_u8[:, 0::2, 0, 1, :]
    d2[:, :, 0, 2, :] = stock_u8[:, 1::2, 0, 0, :]
    d2[:, :, 0, 3, :] = stock_u8[:, 1::2, 0, 1, :]
    return d2


def _ref_muse_prefill(
    q_3d_bf16: np.ndarray,
    kv_new_u8: np.ndarray,
    stock_cache_u8: np.ndarray,
    kv_lens: np.ndarray,
    cu_q_lens: np.ndarray,
    page_indices: np.ndarray,
    *,
    num_active: int,
    pps: int,
    page_size: int,
    sm_scale: float,
) -> np.ndarray:
    """Exact FP32 causal reference for Muse head-packed (`a3_out_layout`) prefill."""
    upd = stock_cache_u8.copy()
    for s in range(num_active):
        qs, qe = int(cu_q_lens[s]), int(cu_q_lens[s + 1])
        ql = qe - qs
        kvl = int(kv_lens[s])
        kv_st = kvl - ql
        for j in range(ql):
            pos = kv_st + j
            pg = int(page_indices[s * pps + pos // page_size])
            upd[pg, pos % page_size, 0, 0, :] = kv_new_u8[qs + j, 0, 0, :]
            upd[pg, pos % page_size, 0, 1, :] = kv_new_u8[qs + j, 0, 1, :]
            upd[pg, pos % page_size, 0, 2:, :] = 0

    upd_f32 = np.asarray(_u8_to_fp8(upd).astype(jnp.float32))
    q_f32 = np.asarray(jnp.asarray(q_3d_bf16, dtype=jnp.bfloat16).astype(jnp.float32))
    out_2d = np.zeros((q_3d_bf16.shape[0], 1024), dtype=np.float32)

    for s in range(num_active):
        qs, qe = int(cu_q_lens[s]), int(cu_q_lens[s + 1])
        ql = qe - qs
        kvl = int(kv_lens[s])
        kv_st = kvl - ql
        k_seq = np.zeros((kvl, 128), dtype=np.float32)
        v_seq = np.zeros((kvl, 128), dtype=np.float32)
        for pos in range(kvl):
            pg = int(page_indices[s * pps + pos // page_size])
            off = pos % page_size
            k_seq[pos] = upd_f32[pg, off, 0, 0, :]
            v_seq[pos] = upd_f32[pg, off, 0, 1, :]

        q_pos = np.arange(kv_st, kvl)[:, None]
        k_pos = np.arange(kvl)[None, :]
        causal_ok = q_pos >= k_pos

        for kv_h in range(2):
            sl = slice(0, 64) if kv_h == 0 else slice(64, 128)
            k_h = k_seq[:, sl]
            v_h = v_seq[:, sl]
            for gp in range(8):
                h = kv_h * 8 + gp
                q_h = q_f32[qs:qe, h, sl]
                scores = (q_h @ k_h.T) * sm_scale
                scores = np.where(causal_ok, scores, -1.0e30)
                m = np.max(scores, axis=-1, keepdims=True)
                p = np.exp(scores - m)
                p /= np.sum(p, axis=-1, keepdims=True)
                out_2d[qs:qe, h * 64 : (h + 1) * 64] = p @ v_h
    return out_2d


class TestMuseLcRpaPrefill(unittest.TestCase):
    def test_default_off_and_no_bare_envfield_bool(self):
        from sglang_tpu.envs import envs

        self.assertEqual(envs.SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS.get(), "")
        for rel in _TOUCHED_FILES:
            src = (_REPO_ROOT / rel).read_text()
            tree = ast.parse(src, filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "value":
                    pass
                # Check envs.SGLANG_TPU_* is never used directly as an `if` test without `.get()`.
                if isinstance(node, ast.If) and isinstance(node.test, ast.Attribute):
                    if isinstance(node.test.value, ast.Name) and node.test.value.id in (
                        "envs",
                        "tpu_envs",
                    ):
                        self.fail(f"Bare EnvField truthiness in {rel}:{node.lineno}")

    def test_lc_opts_routing(self):
        import os
        from sglang_tpu.attention_pallas_rpa_v3 import _invoke_v3_kernel

        captured = {}

        def _fake_rpa(q, k, v, c, kl, pi, cq, ckv, d, cm, sink, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return q, c

        class _FakeArr:
            def __init__(self, shape, dtype="bfloat16"):
                self.shape = shape
                self.dtype = dtype
                self.ndim = len(shape)

        env = {
            "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
            "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS": "bq2048,bkv8192,s2,prefix_fast,gqa4_vreg,fp8_qk",
        }
        with patch.dict(os.environ, env, clear=False):
            _invoke_v3_kernel(
                _fake_rpa,
                queries=_FakeArr((1, 8192, 8, 2, 128), "bfloat16"),
                keys=_FakeArr((8192, 1, 4, 128), "float8_e4m3fn"),
                values=None,
                kv_cache_fused=_FakeArr((1000, 256, 1, 4, 128), "float8_e4m3fn"),
                kv_lens=_FakeArr((8,), "int32"),
                page_indices=_FakeArr((8 * 1089,), "int32"),
                cu_q_lens=_FakeArr((9,), "int32"),
                cu_kv_lens=_FakeArr((9,), "int32"),
                distribution=_FakeArr((3,), "int32"),
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=None,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
                prepared_inputs=True,
                skip_decode=True,
                a3_out_layout=True,
            )
        self.assertEqual(captured["m_block_sizes"], (2048, 8192, 256, 256))
        opts = captured["_prefill_opts"]
        self.assertNotIn("mixed_s3", opts)
        for tok in ("prefix_fast", "gqa4_vreg", "fp8_qk", "bkv_full_unroll"):
            self.assertIn(tok, opts)

        # page_size=256 + KV_PACK_DENSE2=1 (physical page rows = 128) must detect _kv_pack_dense2=True.
        with patch.dict(os.environ, {**env, "SGLANG_TPU_KV_PACK_DENSE2": "1"}, clear=False):
            _invoke_v3_kernel(
                _fake_rpa,
                queries=_FakeArr((1, 8192, 8, 2, 128), "bfloat16"),
                keys=_FakeArr((8192, 1, 4, 128), "float8_e4m3fn"),
                values=None,
                kv_cache_fused=_FakeArr((1000, 128, 1, 4, 128), "float8_e4m3fn"),
                kv_lens=_FakeArr((8,), "int32"),
                page_indices=_FakeArr((8 * 1089,), "int32"),
                cu_q_lens=_FakeArr((9,), "int32"),
                cu_kv_lens=_FakeArr((9,), "int32"),
                distribution=_FakeArr((3,), "int32"),
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=None,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
                prepared_inputs=True,
                skip_decode=True,
                a3_out_layout=True,
            )
        self.assertTrue(captured.get("_kv_pack_dense2", False))

        # SWA layer ignores Full-Attn LC block sizes & prefix_fast.
        with patch.dict(os.environ, env, clear=False):
            _invoke_v3_kernel(
                _fake_rpa,
                queries=_FakeArr((1, 8192, 8, 2, 128), "bfloat16"),
                keys=_FakeArr((8192, 1, 4, 128), "float8_e4m3fn"),
                values=None,
                kv_cache_fused=_FakeArr((1000, 256, 1, 4, 128), "float8_e4m3fn"),
                kv_lens=_FakeArr((8,), "int32"),
                page_indices=_FakeArr((8 * 1089,), "int32"),
                cu_q_lens=_FakeArr((9,), "int32"),
                cu_kv_lens=_FakeArr((9,), "int32"),
                distribution=_FakeArr((3,), "int32"),
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=2049,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
                prepared_inputs=True,
                skip_decode=True,
                a3_out_layout=True,
            )
        self.assertEqual(captured["m_block_sizes"], (256, 2048, 256, 256))
        self.assertNotIn("prefix_fast", captured["_prefill_opts"])
        self.assertNotIn("staged_qo", captured["_prefill_opts"])
        self.assertIn("mixed_s3", captured["_prefill_opts"])

        # bq8192 auto-enables staged_qo and falls back to bq4096 when T=4096.
        env_8k = {
            "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_ring_fuse",
            "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS": "bq8192,s2,prefix_fast,fp8_qk,vreg_acc",
        }
        for q_tok, exp_bq in ((8192, 8192), (4096, 4096)):
            with patch.dict(os.environ, env_8k, clear=False):
                _invoke_v3_kernel(
                    _fake_rpa,
                    queries=_FakeArr((1, q_tok, 8, 2, 128), "bfloat16"),
                    keys=_FakeArr((q_tok, 1, 4, 128), "float8_e4m3fn"),
                    values=None,
                    kv_cache_fused=_FakeArr((1000, 256, 1, 4, 128), "float8_e4m3fn"),
                    kv_lens=_FakeArr((8,), "int32"),
                    page_indices=_FakeArr((8 * 1089,), "int32"),
                    cu_q_lens=_FakeArr((9,), "int32"),
                    cu_kv_lens=_FakeArr((9,), "int32"),
                    distribution=_FakeArr((3,), "int32"),
                    attention_sink=None,
                    sm_scale=0.125,
                    sliding_window=None,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    prepared_inputs=True,
                    skip_decode=True,
                    a3_out_layout=True,
                )
            self.assertEqual(captured["m_block_sizes"], (exp_bq, 4096, 256, 256))
            self.assertIn("staged_qo", captured["_prefill_opts"])
            self.assertIn("vreg_acc", captured["_prefill_opts"])

        with patch.dict(
            os.environ,
            {
                **env_8k,
                "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS": "bq4096,s2,prefix_fast,staged_qo,fp8_qkpv_dyn",
            },
            clear=False,
        ):
            _invoke_v3_kernel(
                _fake_rpa,
                queries=_FakeArr((1, 8192, 8, 2, 128), "bfloat16"),
                keys=_FakeArr((8192, 1, 4, 128), "float8_e4m3fn"),
                values=None,
                kv_cache_fused=_FakeArr((1000, 256, 1, 4, 128), "float8_e4m3fn"),
                kv_lens=_FakeArr((8,), "int32"),
                page_indices=_FakeArr((8 * 1089,), "int32"),
                cu_q_lens=_FakeArr((9,), "int32"),
                cu_kv_lens=_FakeArr((9,), "int32"),
                distribution=_FakeArr((3,), "int32"),
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=None,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
                prepared_inputs=True,
                skip_decode=True,
                a3_out_layout=True,
            )
        self.assertIn("fp8_qkpv_dyn", captured["_prefill_opts"])

        with patch.dict(
            os.environ,
            {
                **env_8k,
                "SGLANG_TPU_RPA_V3_PREFILL_LC_OPTS": "bq4096,s2,prefix_fast,staged_qo,fp8_qkpv_fast,vreg_acc1,mrb1",
            },
            clear=False,
        ):
            _invoke_v3_kernel(
                _fake_rpa,
                queries=_FakeArr((1, 8192, 8, 2, 128), "bfloat16"),
                keys=_FakeArr((8192, 1, 4, 128), "float8_e4m3fn"),
                values=None,
                kv_cache_fused=_FakeArr((1000, 256, 1, 4, 128), "float8_e4m3fn"),
                kv_lens=_FakeArr((8,), "int32"),
                page_indices=_FakeArr((8 * 1089,), "int32"),
                cu_q_lens=_FakeArr((9,), "int32"),
                cu_kv_lens=_FakeArr((9,), "int32"),
                distribution=_FakeArr((3,), "int32"),
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=None,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
                prepared_inputs=True,
                skip_decode=True,
                a3_out_layout=True,
            )
        for tok in ("fp8_qkpv_fast", "vreg_acc1", "mrb1"):
            self.assertIn(tok, captured["_prefill_opts"])

    def test_shared_pages_bs8_dispatch_gate(self):
        from sglang_tpu.attention.pallas import rpa_v3_decode_fast
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )

        seen = []

        def _fake_run_decode(*args, **kwargs):
            seen.append(kwargs.get("shared_pages"))
            return args[0], args[3]

        for bs, expect_non_none in ((8, True), (4, False)):
            seen.clear()
            with (
                patch.object(rpa_v3_decode_fast, "can_use_decode_fast", return_value=True),
                patch.object(rpa_v3_decode_fast, "run_decode_fast_rpa", side_effect=_fake_run_decode),
            ):
                ragged_paged_attention(
                    jnp.zeros((1, bs, 8, 2, 128), dtype=jnp.bfloat16),
                    jnp.zeros((bs, 1, 4, 128), dtype=jnp.float8_e4m3fn),
                    None,
                    jnp.zeros((16, 128, 1, 4, 128), dtype=jnp.float8_e4m3fn),
                    jnp.full((bs,), 1024, dtype=jnp.int32),
                    jnp.zeros((bs * 129,), dtype=jnp.int32),
                    jnp.arange(bs + 1, dtype=jnp.int32),
                    jnp.arange(bs + 1, dtype=jnp.int32) * 1024,
                    jnp.array([bs, bs, bs, 2], dtype=jnp.int32),
                    None,
                    None,
                    vmem_limit_bytes=125829120,
                    _decode_fast="shared64,c4096,q512,pbf16,w8",
                    _skip_mixed=True,
                    _prepared_inputs=True,
                    _kv_pack_dense2=True,
                )
            self.assertEqual(len(seen), 1)
            if expect_non_none:
                self.assertIsNotNone(seen[0])
            else:
                self.assertIsNone(seen[0])

    def test_fp8_qkpv_dyn_math_parity(self):
        rng = np.random.default_rng(1337)
        w_cols = 256
        n_slices = 4
        q_mat = rng.standard_normal((w_cols, 64)).astype(np.float32) * 0.4
        # Inject hot queries (+25 logit scale), normal queries, and extreme negative queries (-150).
        q_mat[:32] *= 18.0
        q_mat[32:64] *= 0.05
        k_slices = [rng.standard_normal((256, 64)).astype(np.float32) * 0.4 for _ in range(n_slices)]
        v_slices = [rng.standard_normal((256, 64)).astype(np.float32) * 0.4 for _ in range(n_slices)]
        # Make slice 1 peak high on rows 0..31 and slice 2 negative.
        k_slices[1][:64] += 1.8
        k_slices[2] -= 1.2
        k_f8 = [jnp.asarray(k, dtype=jnp.float8_e4m3fn) for k in k_slices]
        v_f8 = [jnp.asarray(v, dtype=jnp.float8_e4m3fn) for v in v_slices]
        q_f8 = jnp.asarray(jnp.clip(q_mat, -448.0, 448.0), dtype=jnp.float8_e4m3fn)
        exp_scale = np.float32(0.125 * np.log2(np.e))

        # Exact FP32 reference using the same quantized FP8 inputs (isolates fp8_qkpv_dyn P quantization).
        k_all = np.concatenate([np.asarray(k.astype(jnp.float32)) for k in k_f8], axis=0)
        v_all = np.concatenate([np.asarray(v.astype(jnp.float32)) for v in v_f8], axis=0)
        q_in = np.asarray(q_f8.astype(jnp.float32))
        logits = (q_in @ k_all.T) * np.float32(0.125)
        # Mask out slice 0 completely for rows 64..127 and half of slice 3 causally.
        logits[64:128, :256] = -np.inf
        logits[:, 3 * 256 + 128 :] = -np.inf
        # Also test extreme negative slices (-80 and -150 in natural-log units -> -115 and -216 in log2) on rows 32..63.
        logits[32:48, : 3 * 256] = -80.0
        logits[48:64, : 3 * 256] = -150.0
        m_ref = np.max(logits, axis=1, keepdims=True)
        w_ref = np.exp(logits - m_ref)
        ref = (w_ref @ v_all) / np.sum(w_ref, axis=1, keepdims=True)

        acc = jnp.zeros((64, w_cols), dtype=jnp.float32)
        l_acc = jnp.zeros((w_cols,), dtype=jnp.float32)
        for s_idx in range(n_slices):
            qk = jnp.asarray(k_f8[s_idx], dtype=jnp.float32) @ jnp.asarray(q_f8, dtype=jnp.float32).T
            qk_sc = qk * exp_scale
            if s_idx == 0:
                qk_sc = qk_sc.at[:, 64:128].set(-jnp.inf)
            if s_idx == 3:
                qk_sc = qk_sc.at[128:, :].set(-jnp.inf)
            if s_idx < 3:
                qk_sc = qk_sc.at[:, 32:48].set(-80.0 * float(np.log2(np.e)))
                qk_sc = qk_sc.at[:, 48:64].set(-150.0 * float(np.log2(np.e)))
            m_col = jnp.max(qk_sc, axis=0, keepdims=True)
            m_safe = jnp.maximum(m_col, jnp.float32(-110.0))
            m_bias = m_safe - jnp.float32(8.75)
            p_f32 = jnp.exp2(qk_sc - m_bias)
            p_f8 = jnp.minimum(p_f32, jnp.float32(448.0)).astype(jnp.float8_e4m3fn)
            p_scale = jnp.exp2(m_bias)[0]
            l_acc = l_acc + jnp.sum(p_f8.astype(jnp.float32), axis=0) * p_scale
            vp = jnp.asarray(v_f8[s_idx], dtype=jnp.float32).T @ p_f8.astype(jnp.float32)
            acc = acc + vp * p_scale[None, :]
        got = np.asarray((acc / l_acc[None, :]).T)
        self.assertFalse(np.isnan(got).any())
        cos = float(np.sum(got * ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
        self.assertGreater(cos, 0.999)
        np.testing.assert_allclose(got, ref, rtol=3e-2, atol=3e-2)

        # Verify fp8_qkpv_fast (exp_scale folded into Q, bf16 max/sub/exp2/psum).
        q_scaled_f8 = jnp.asarray(
            jnp.clip(q_mat * float(exp_scale), -448.0, 448.0),
            dtype=jnp.float8_e4m3fn,
        )
        acc_f = jnp.zeros((64, w_cols), dtype=jnp.float32)
        l_f = jnp.zeros((w_cols,), dtype=jnp.float32)
        for s_idx in range(n_slices):
            qk_bf = (
                jnp.asarray(k_f8[s_idx], dtype=jnp.float32)
                @ jnp.asarray(q_scaled_f8, dtype=jnp.float32).T
            ).astype(jnp.bfloat16)
            if s_idx == 0:
                qk_bf = qk_bf.at[:, 64:128].set(jnp.bfloat16(-jnp.inf))
            if s_idx == 3:
                qk_bf = qk_bf.at[128:, :].set(jnp.bfloat16(-jnp.inf))
            if s_idx < 3:
                qk_bf = qk_bf.at[:, 32:48].set(jnp.bfloat16(-80.0 * float(np.log2(np.e))))
                qk_bf = qk_bf.at[:, 48:64].set(jnp.bfloat16(-150.0 * float(np.log2(np.e))))
            m_col_bf = jnp.max(qk_bf, axis=0, keepdims=True)
            m_safe_bf = jnp.maximum(m_col_bf, jnp.bfloat16(-110.0))
            m_bias_bf = m_safe_bf - jnp.bfloat16(8.75)
            p_f8_fast = jnp.minimum(
                jnp.exp2(qk_bf - m_bias_bf), jnp.bfloat16(448.0)
            ).astype(jnp.float8_e4m3fn)
            p_scale_f = jnp.exp2(m_bias_bf.astype(jnp.float32))[0]
            l_f = l_f + jnp.sum(p_f8_fast.astype(jnp.bfloat16), axis=0).astype(jnp.float32) * p_scale_f
            vp_f = jnp.asarray(v_f8[s_idx], dtype=jnp.float32).T @ p_f8_fast.astype(jnp.float32)
            acc_f = acc_f + vp_f * p_scale_f[None, :]
        got_fast = np.asarray((acc_f / l_f[None, :]).T)
        self.assertFalse(np.isnan(got_fast).any())
        cos_fast = float(np.sum(got_fast * ref) / (np.linalg.norm(got_fast) * np.linalg.norm(ref)))
        self.assertGreater(cos_fast, 0.997)

    def test_streaming_prefix_fast_and_gqa4_tpu(self):
        if jax.devices()[0].platform != "tpu":
            self.skipTest("stationary_q_v2 uses Ref.bitcast (Mosaic TPU only; unsupported in CPU interpret)")
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            _ragged_paged_attention_kernel,
        )

        # Force streaming fallback (pps * page_size > 32768) in _stationary_q_v2_process
        # by testing directly with prefix_fast + gqa4_vreg + bq_sz=1024 + bkv_sz=1024
        # on CPU interpret=True with detect_races=True.
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            RpaCase,
        )
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        rng = np.random.default_rng(2026)
        for page_size, pps, use_d2, extra_opts in (
            (128, 257, False, {"gqa4_vreg", "fp8_qk", "prefix_fast"}),
            (256, 129, False, {"gqa4_vreg", "fp8_qk_dyn", "f32_exp", "prefix_fast"}),
            (256, 129, True, {"gqa4_vreg", "fp8_qk", "prefix_fast"}),
            (256, 129, True, {"vreg_acc", "fp8_qkpv_dyn", "prefix_fast"}),
        ):
            with self.subTest(page_size=page_size, use_d2=use_d2, opts=sorted(extra_opts)):
                b_sz = 2
                q_lens = np.array([640, 384], dtype=np.int32)
                pref_lens = np.array([1536, 1024], dtype=np.int32)
                kv_lens = pref_lens + q_lens
                cu_q_lens = np.array([0, 640, 1024], dtype=np.int32)
                cu_kv_lens = np.arange(b_sz + 1, dtype=np.int32) * (pps * page_size)
                dist = np.array([0, 0, 2], dtype=np.int32)
                # Use dense logical page table over only the needed pages in a compact backing store
                # mapped cleanly modulo n_pages? Better: allocate exact distinct pages needed.
                page_indices = np.zeros((b_sz * pps,), dtype=np.int32)
                next_pg = 1
                for s in range(b_sz):
                    n_p = (int(kv_lens[s]) + page_size - 1) // page_size
                    page_indices[s * pps : s * pps + n_p] = np.arange(
                        next_pg, next_pg + n_p, dtype=np.int32
                    )
                    next_pg += n_p
                n_pages = next_pg

                stock_u8 = np.full((n_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
                for s in range(b_sz):
                    for pos in range(int(pref_lens[s])):
                        pg = int(page_indices[s * pps + pos // page_size])
                        off = pos % page_size
                        stock_u8[pg, off, 0, :2, :] = _rand_fp8_u8(
                            rng, (2, 128), scale=0.35
                        )
                        stock_u8[pg, off, 0, 2:, :] = 0
                cache_u8 = (
                    _pack_stock_cache_to_dense2_u8(stock_u8) if use_d2 else stock_u8
                )

                t_tot = 1024
                q_3d = _rand_muse_q_3d(rng, t_tot, scale=0.3)
                kv_new_u8 = np.zeros((t_tot, 1, 4, 128), dtype=np.uint8)
                kv_new_u8[:, 0, :2, :] = _rand_fp8_u8(rng, (t_tot, 2, 128), scale=0.35)

                ref_out = _ref_muse_prefill(
                    q_3d,
                    kv_new_u8,
                    stock_u8,
                    kv_lens,
                    cu_q_lens,
                    page_indices,
                    num_active=2,
                    pps=pps,
                    page_size=page_size,
                    sm_scale=0.125,
                )

                bq_sz, bkv_sz = 1024, 512
                opts = frozenset({"stationary_q_m0_ring_fuse", "skip_empty_cdma", "bkv_full_unroll"} | extra_opts)
                use_fp8_qk = "fp8_qk" in opts or "fp8_qk_dyn" in opts or "fp8_qkpv_dyn" in opts
                use_vt_fp8 = "fp8_qkpv_dyn" in opts
                use_fp8_dyn = "fp8_qk_dyn" in opts
                num_slots = 2
                phys_pg = page_size // 2 if use_d2 else page_size
                bkv_phys = bkv_sz // 2 if use_d2 else bkv_sz
                extra_scratches = [
                    pltpu.VMEM((16, 64, bq_sz), jnp.float8_e4m3fn if use_fp8_qk else jnp.bfloat16),
                    pltpu.VMEM((2, bkv_sz // 256, 256, 128), jnp.float8_e4m3fn if use_fp8_qk else jnp.bfloat16),
                    pltpu.VMEM((2, bkv_sz // 256, 64, 256), jnp.float8_e4m3fn if use_vt_fp8 else jnp.bfloat16),
                    pltpu.VMEM((1, 256, 256), jnp.bfloat16),
                    None,
                    pltpu.VMEM((16, 1024), jnp.bfloat16),
                    pltpu.VMEM((16, 8, bq_sz), jnp.float32) if use_fp8_dyn else None,
                ]
                if use_d2:
                    extra_scratches.append(
                        pltpu.VMEM((num_slots, bkv_sz, 1, 4, 128), jnp.float8_e4m3fn)
                    )
                scratch_shapes = [
                    None,
                    pltpu.VMEM((num_slots, bkv_phys, 1, 4, 128), jnp.float8_e4m3fn),
                    pltpu.VMEM((num_slots, 1, bq_sz, 8, 2, 128), jnp.bfloat16),
                    pltpu.VMEM((num_slots, bq_sz + 16, 1024), jnp.bfloat16),
                    pltpu.SemaphoreType.DMA((6 if use_d2 else 5, num_slots)),
                    pltpu.VMEM((16, 8, bq_sz), jnp.float32),
                    None,
                    pltpu.VMEM((16, 64, bq_sz), jnp.float32),
                    *extra_scratches,
                ]
                import functools

                kernel = pl.pallas_call(
                    functools.partial(
                        _ragged_paged_attention_kernel,
                        causal=True,
                        sm_scale=0.125,
                        sliding_window=None,
                        soft_cap=None,
                        mask_value=-0.7 * float(jnp.finfo(jnp.dtype("float32")).max),
                        q_scale=None,
                        k_scale=1.0,
                        v_scale=1.0,
                        xai_temperature_len=None,
                        static_q_len=None,
                        bq_sz=bq_sz,
                        bkv_sz=bkv_sz,
                        bq_csz=256,
                        bkv_csz=256,
                        case=RpaCase.MIXED,
                        prefill_opts=opts | {"a3_out_layout"},
                        kv_pack_dense2=use_d2,
                    ),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=9,
                        in_specs=[
                            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                            None,
                            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                            None,
                        ],
                        out_specs=[
                            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
                        ],
                        grid=(1,),
                        scratch_shapes=scratch_shapes,
                    ),
                    out_shape=[
                        jax.ShapeDtypeStruct((t_tot, 1024), jnp.bfloat16),
                        jax.ShapeDtypeStruct((n_pages, phys_pg, 1, 4, 128), jnp.float8_e4m3fn),
                    ],
                    input_output_aliases={11: 1},
                    interpret=pltpu.InterpretParams(
                        detect_races=True, uninitialized_memory="nan"
                    ),
                )
                out_k, _ = kernel(
                    jnp.asarray(kv_lens),
                    jnp.asarray(page_indices),
                    jnp.asarray(cu_q_lens),
                    jnp.asarray(cu_kv_lens),
                    jnp.array([0], dtype=jnp.int32),
                    jnp.asarray(dist),
                    jnp.zeros((4,), dtype=jnp.int32),
                    jnp.full((2 * num_slots,), -1, dtype=jnp.int32),
                    jnp.full((3 * num_slots,), -1, dtype=jnp.int32),
                    _pack_q_5d(q_3d),
                    _u8_to_fp8(kv_new_u8),
                    _u8_to_fp8(cache_u8),
                    None,
                    jnp.zeros((bkv_sz, 128), dtype=jnp.int32),
                    None,
                )
                got = np.asarray(out_k.astype(jnp.float32))
                self.assertFalse(np.isnan(got).any())
                np.testing.assert_allclose(got, ref_out, rtol=3e-2, atol=3e-2)

    def test_tpu7x_mosaic_exports_lc_prefill(self):
        try:
            from jax._src import tpu_info as _tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available in lightweight CPU venv")

        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        base_fl = {
            "stationary_q_m0_ring_fuse",
            "skip_empty_cdma",
            "bkv_full_unroll",
            "prefix_fast",
        }
        cases = (
            ("lc_bq2048_bkv4k_s2_gqa4_fp8qk_p128", 8192, 128, 2177, (2048, 4096, 256, 256), base_fl | {"gqa4_vreg", "fp8_qk"}, False, True),
            ("lc_bq2048_bkv4k_s2_gqa4_fp8qk_p256", 8192, 256, 1089, (2048, 4096, 256, 256), base_fl | {"gqa4_vreg", "fp8_qk"}, False, True),
            ("lc_bq2048_bkv4k_s2_gqa4_fp8qk_p256_d2", 8192, 256, 1089, (2048, 4096, 256, 256), base_fl | {"gqa4_vreg", "fp8_qk"}, True, True),
            ("lc_bq2048_bkv8k_s2_gqa2_fp8qk_p256", 8192, 256, 1089, (2048, 8192, 256, 256), base_fl | {"gqa2_vreg", "fp8_qk"}, False, True),
            ("lc_bq4096_bkv4k_staged_gqa1_fp8qk_vregacc_p256_d2", 8192, 256, 1089, (4096, 4096, 256, 256), base_fl | {"staged_qo", "fp8_qk", "vreg_acc"}, True, True),
            ("lc_bq4096_bkv8k_staged_gqa1_fp8qk_p256_d2", 16384, 256, 1089, (4096, 8192, 256, 256), base_fl | {"staged_qo", "fp8_qk"}, True, True),
            ("lc_bq8192_bkv4k_staged_gqa1_fp8qk_p256_d2", 16384, 256, 1089, (8192, 4096, 256, 256), base_fl | {"staged_qo", "fp8_qk"}, True, True),
            ("lc_bq4096_bkv4k_staged_nona3_p128_d2", 8192, 128, 2177, (4096, 4096, 256, 256), base_fl | {"staged_qo", "fp8_qk"}, True, False),
            ("lc_bq2048_bkv4k_s2_gqa4_fp8qk_dyn_p256", 8192, 256, 1089, (2048, 4096, 256, 256), base_fl | {"gqa4_vreg", "fp8_qk_dyn", "f32_exp"}, False, True),
            ("lc_bq4096_bkv4k_staged_gqa1_fp8qkpvdyn_p256_d2", 8192, 256, 1089, (4096, 4096, 256, 256), base_fl | {"staged_qo", "fp8_qkpv_dyn"}, True, True),
            ("lc_bq4096_bkv4k_staged_vregacc_fp8qkpvdyn_p256_d2", 8192, 256, 1089, (4096, 4096, 256, 256), base_fl | {"staged_qo", "vreg_acc", "fp8_qkpv_dyn"}, True, True),
            ("lc_bq2048_bkv4k_gqa2_fp8qkpvdyn_p256_d2", 8192, 256, 1089, (2048, 4096, 256, 256), base_fl | {"gqa2_vreg", "fp8_qkpv_dyn"}, True, True),
            ("lc_bq4096_bkv4k_vregacc_fp8qkpvfast_p256_d2", 8192, 256, 1089, (4096, 4096, 256, 256), base_fl | {"staged_qo", "vreg_acc", "fp8_qkpv_fast"}, True, True),
            ("lc_bq8192_bkv4k_vregacc1_mrb1_fp8qkpvfast_p256_d2", 8192, 256, 1089, (8192, 4096, 256, 256), base_fl | {"staged_qo", "vreg_acc1", "mrb1", "fp8_qkpv_fast"}, True, True),
            ("lc_bq4096_bkv8k_vregacc1_mrb1_fp8qk_p256_d2", 8192, 256, 1089, (4096, 8192, 256, 256), base_fl | {"staged_qo", "vreg_acc1", "mrb1", "fp8_qk"}, True, True),
        )
        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            for tag, t_tok, pg_sz, pps, mb, opts, use_d2, use_a3 in cases:
                with self.subTest(export=tag):
                    phys_pg = pg_sz // 2 if use_d2 else pg_sz
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q, kv, c, kl, pi, cq, ckv, d, _mb=mb, _op=frozenset(opts), _d2=use_d2, _a3=use_a3: ragged_paged_attention(
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
                                sliding_window=None,
                                k_scale=1.0,
                                v_scale=1.0,
                                m_block_sizes=_mb,
                                _prefill_opts=_op,
                                _fused_out_pack=2,
                                _prepared_inputs=True,
                                _skip_decode=True,
                                _a3_out_layout=_a3,
                                _kv_pack_dense2=_d2,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, t_tok, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((t_tok, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((4096, phys_pg, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((64,), jnp.int32),
                        jax.ShapeDtypeStruct((64 * pps,), jnp.int32),
                        jax.ShapeDtypeStruct((65,), jnp.int32),
                        jax.ShapeDtypeStruct((65,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("RPAm", hlo)


if __name__ == "__main__":
    unittest.main()
