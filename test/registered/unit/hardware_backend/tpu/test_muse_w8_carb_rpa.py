"""Wave-8 RPA v3 (`carb-k-rpa`) unit, CPU `interpret=True`, and TPU7x Mosaic export tests.

Covers:
1. Default-OFF flag preservation & zero bare `EnvField` boolean truthiness across touched files.
2. `SGLANG_TPU_KV_PACK_DENSE2=1` (`TpuFusedMHATokenToKVPool`): `(N, 64, 1, 4, 128)` physical layout,
   `256 B/tok` pricing (`50%` KV HBM & DMA reduction), `_logical_kv_view`, and draft-pool exclusion.
3. `RPAd_fast` (`rpa_v3_decode_fast.py`):
   - `kv_pack_dense2=True` CPU `interpret=True` (`0x7F` NaN-poisoned, `detect_races=True`) across
     even/odd `kv_lens`, multi-step page-boundary transitions (`127 -> 128 -> 129`), SWA & Full,
     verifying bit-identical outputs vs stock `kv_pack_dense2=False` and exact FP32 reference.
   - `w8` (`swafast2` + `hoist_new_kv` + `fine_gears` `c512` + `w7auto` `pbf16`) numerical accuracy.
   - DFlash `TARGET_VERIFY` (`dq=4, 5, 6`) single-MXU-pass `[16*dq, 128]` with in-VMEM `onorm + gate`,
     for both stock and `kv_pack_dense2=True`, including odd cross-page multi-row KV writes.
4. `write_kv_only_rpa_v3(..., kv_pack_dense2=True)` CPU `interpret=True` across ragged even/odd
   `kv_start` and `q_len` (preserving pre-existing even partner when `kv_start` is odd and leaving
   untouched rows un-clobbered).
5. TPU7x Mosaic `jax.export` compilation verification for `RPAd_fast`, `RPAm` (`gqa2_vreg` fix,
   `vreg_acc`, `sw_short_fuse`, `fp8_qkpv_norm`, `fp8_qkpv_s0`, `_kv_pack_dense2=True`), and
   `write_kv_only_rpa_v3(..., kv_pack_dense2=True)`.
"""

from __future__ import annotations

import ast
import os
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
    "sglang_tpu/kv_pool.py",
    "sglang_tpu/hooks/draft_kv_pool.py",
    "sglang_tpu/dflash_draft_kv_write.py",
    "sglang_tpu/unified_kv_pool.py",
    "sglang_tpu/attention_pallas_rpa_v3.py",
    "sglang_tpu/attention/pallas/rpa_v3_decode_fast.py",
    "sglang_tpu/attention/pallas/ragged_paged_attention_v3.py",
)


def _u8_to_fp8(arr_u8: np.ndarray) -> jax.Array:
    return jax.lax.bitcast_convert_type(jnp.asarray(arr_u8, dtype=jnp.uint8), jnp.float8_e4m3fn)


def _fp8_to_u8(arr_f8: jax.Array) -> np.ndarray:
    return np.asarray(jax.lax.bitcast_convert_type(arr_f8, jnp.uint8))


def _rand_fp8_u8(rng: np.random.Generator, shape: tuple[int, ...], scale: float = 0.4) -> np.ndarray:
    """Sample realistic finite FP8 e4m3fn values (`O(scale)`) and return raw uint8 bytes."""
    vals = (rng.standard_normal(shape) * scale).astype(np.float32)
    return _fp8_to_u8(jnp.asarray(vals, dtype=jnp.float8_e4m3fn))


def _rand_muse_q_3d(rng: np.random.Generator, n_tok: int, scale: float = 0.35) -> np.ndarray:
    """Muse head-packed `q_3d [T, 16, 128]` has heads 0..7 in `[:64]` and 8..15 in `[64:]`."""
    q = (rng.standard_normal((n_tok, 16, 128)) * scale).astype(np.float32)
    q[:, :8, 64:] = 0.0
    q[:, 8:, :64] = 0.0
    return q


def _pack_q_5d(q_3d: np.ndarray) -> jax.Array:
    t = q_3d.shape[0]
    return jnp.asarray(q_3d, dtype=jnp.bfloat16).reshape(1, t, 8, 2, 128)


def _pack_stock_cache_to_dense2_u8(stock_u8: np.ndarray) -> np.ndarray:
    """Convert stock `(N, 128, 1, 4, 128)` u8 (with bytes 0=K, 1=V) into dense2 `(N, 64, 1, 4, 128)` u8."""
    n_pg = stock_u8.shape[0]
    d2 = np.full((n_pg, 64, 1, 4, 128), 0x7F, dtype=np.uint8)
    d2[:, :, 0, 0, :] = stock_u8[:, 0::2, 0, 0, :]
    d2[:, :, 0, 1, :] = stock_u8[:, 0::2, 0, 1, :]
    d2[:, :, 0, 2, :] = stock_u8[:, 1::2, 0, 0, :]
    d2[:, :, 0, 3, :] = stock_u8[:, 1::2, 0, 1, :]
    return d2


def _unpack_dense2_u8_token(d2_u8: np.ndarray, pg: int, pos_in_pg: int) -> tuple[np.ndarray, np.ndarray]:
    row = pos_in_pg // 2
    if (pos_in_pg & 1) == 0:
        return d2_u8[pg, row, 0, 0, :], d2_u8[pg, row, 0, 1, :]
    return d2_u8[pg, row, 0, 2, :], d2_u8[pg, row, 0, 3, :]


def _merge_interpret_aliased_d2_u8(
    prev_d2_u8: np.ndarray,
    upd_d2_u8: np.ndarray,
    kv_lens: np.ndarray,
    page_indices: np.ndarray,
    *,
    num_active: int,
    dq: int,
    pps: int,
) -> np.ndarray:
    """Apply written `[pair_lo, pair_hi)` rows from CPU `interpret=True` output onto `prev_d2_u8`.

    Note: JAX's Pallas CPU `interpret=True` mode with `uninitialized_memory="nan"` leaves
    non-DMA'd HBM output slices at `0x7F` even when `input_output_aliases` is specified
    (whereas real TPU HBM aliases in-place). Merging only the exact written `[pair_lo, pair_hi)`
    rows also verifies that every written row (including odd-start merged partners) came from DMA!
    """
    merged = prev_d2_u8.copy()
    for s in range(num_active):
        kvl = int(kv_lens[s])
        kv_st = kvl - dq
        p_lo = kv_st // 128
        p_hi = (kvl + 127) // 128
        for p in range(p_lo, p_hi):
            t_lo = max(kv_st, p * 128) - p * 128
            t_hi = min(kvl, (p + 1) * 128) - p * 128
            r_lo = t_lo // 2
            r_hi = (t_hi + 1) // 2
            pg = int(page_indices[s * pps + p])
            merged[pg, r_lo:r_hi] = upd_d2_u8[pg, r_lo:r_hi]
    return merged


def _ref_muse_decode_or_verify(
    q_3d_bf16: np.ndarray,
    kv_new_u8: np.ndarray,
    stock_cache_u8: np.ndarray,
    kv_lens: np.ndarray,
    page_indices: np.ndarray,
    *,
    num_active: int,
    dq: int,
    pps: int,
    sm_scale: float,
    sliding_window: int | None,
    gate_bf16: np.ndarray | None = None,
    out_norm_eps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact FP32 reference for Muse head-packed (`fused_out_pack=2`) decode (`dq=1`) & verify (`dq>=2`)."""
    upd_stock = stock_cache_u8.copy()
    total_rows = q_3d_bf16.shape[0]
    out_2d = np.zeros((total_rows, 1024), dtype=np.float32)

    q_bf16 = np.asarray(jnp.asarray(q_3d_bf16, dtype=jnp.bfloat16).astype(jnp.float32))

    for s in range(num_active):
        kvl = int(kv_lens[s])
        kv_st = kvl - dq
        for j in range(dq):
            pos = kv_st + j
            pg = int(page_indices[s * pps + pos // 128])
            upd_stock[pg, pos % 128, 0, 0, :] = kv_new_u8[s * dq + j, 0, 0, :]
            upd_stock[pg, pos % 128, 0, 1, :] = kv_new_u8[s * dq + j, 0, 1, :]
            upd_stock[pg, pos % 128, 0, 2:, :] = 0

    upd_f32 = np.asarray(_u8_to_fp8(upd_stock).astype(jnp.float32))

    for s in range(num_active):
        kvl = int(kv_lens[s])
        k_seq = np.zeros((kvl, 128), dtype=np.float32)
        v_seq = np.zeros((kvl, 128), dtype=np.float32)
        for pos in range(kvl):
            pg = int(page_indices[s * pps + pos // 128])
            k_seq[pos] = upd_f32[pg, pos % 128, 0, 0, :]
            v_seq[pos] = upd_f32[pg, pos % 128, 0, 1, :]

        for j in range(dq):
            row = s * dq + j
            q_pos = (kvl - dq) + j
            ks = np.arange(kvl, dtype=np.int32)
            vis = ks <= q_pos
            if sliding_window is not None:
                vis = vis & (q_pos - ks < sliding_window)

            parts = []
            for g in range(2):
                q_g = q_bf16[row, g * 8 : (g + 1) * 8, g * 64 : (g + 1) * 64]
                k_g = k_seq[:, g * 64 : (g + 1) * 64]
                v_g = v_seq[:, g * 64 : (g + 1) * 64]
                logits = (q_g @ k_g.T) * sm_scale
                logits = np.where(vis[None, :], logits, -np.inf)
                m = np.max(logits, axis=-1, keepdims=True)
                p = np.exp(logits - m)
                p = np.where(vis[None, :], p, 0.0)
                denom = np.sum(p, axis=-1, keepdims=True)
                o_g = (p @ v_g) / denom
                parts.append(o_g.reshape(512))
            out_row = np.concatenate(parts, axis=-1)
            if out_norm_eps is not None:
                heads = out_row.reshape(16, 64)
                inv_r = 1.0 / np.sqrt(np.mean(heads * heads, axis=-1, keepdims=True) + out_norm_eps)
                out_row = (heads * inv_r).reshape(1024)
            if gate_bf16 is not None:
                g_f = np.asarray(jnp.asarray(gate_bf16[row], dtype=jnp.bfloat16).astype(jnp.float32))
                sig = 1.0 / (1.0 + np.exp(-g_f))
                out_row = sig * out_row
            out_2d[row] = out_row

    return out_2d, upd_stock


class TestMuseW8CarbRpa(unittest.TestCase):
    def test_default_off_flags_wiring_and_no_bare_envfield_bool_eval(self):
        from sglang_tpu.envs import envs

        for env_key in (
            "SGLANG_TPU_KV_PACK_DENSE2",
            "SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY",
            "SGLANG_TPU_RPA_V3_DECODE_FAST_W8",
        ):
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop(env_key, None)
                field = getattr(envs, env_key)
                self.assertIs(field.get(), False, f"{env_key} must default to False")

        # Static AST audit: forbid `if envs.SGLANG_...:` without `.get()` in all touched files.
        for rel in _TOUCHED_FILES:
            src = (_REPO_ROOT / rel).read_text(encoding="utf-8")
            tree = ast.parse(src, filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.IfExp, ast.While)):
                    tests = [node.test]
                elif isinstance(node, ast.BoolOp):
                    tests = list(node.values)
                elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                    tests = [node.operand]
                else:
                    continue
                for t in tests:
                    if (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "envs"
                        and t.attr.startswith("SGLANG_")
                    ):
                        self.fail(f"Bare EnvField truthiness evaluation in {rel}:{t.lineno}: envs.{t.attr}")

        # If torch is available, also verify `_invoke_v3_kernel` kwargs routing directly.
        try:
            import torch  # noqa: F401
        except ImportError:
            return

        from sglang_tpu.attention_pallas_rpa_v3 import _invoke_v3_kernel

        captured = {}

        def _fake_rpa(*args, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return "ok"

        q_pref = jax.ShapeDtypeStruct((1, 1024, 8, 2, 128), jnp.bfloat16)
        kv_pref = jax.ShapeDtypeStruct((1024, 1, 4, 128), jnp.float8_e4m3fn)
        c_d2 = jax.ShapeDtypeStruct((100, 64, 1, 4, 128), jnp.float8_e4m3fn)
        kl = jax.ShapeDtypeStruct((4,), jnp.int32)
        pi = jax.ShapeDtypeStruct((4 * 17,), jnp.int32)
        cq = jax.ShapeDtypeStruct((5,), jnp.int32)
        dist = jax.ShapeDtypeStruct((3,), jnp.int32)

        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_KV_PACK_DENSE2": "1",
                "SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE": "m0_f32e_fuse",
                "SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS": "w8,fp8_qkpv_norm",
                "SGLANG_TPU_RPA_V3_PREFILL_SW2048_FUSE": "1",
            },
            clear=False,
        ):
            _invoke_v3_kernel(
                _fake_rpa,
                queries=q_pref,
                keys=kv_pref,
                values=None,
                kv_cache_fused=c_d2,
                kv_lens=kl,
                page_indices=pi,
                cu_q_lens=cq,
                cu_kv_lens=cq,
                distribution=dist,
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=2048,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
                prepared_inputs=True,
                skip_decode=True,
                a3_out_layout=2,
            )
            self.assertTrue(captured.get("_kv_pack_dense2"))
            popts = captured["_prefill_opts"]
            self.assertIn("vreg_acc", popts)
            self.assertIn("sw_short_fuse", popts)
            self.assertIn("fp8_qkpv_norm", popts)
            # Ensure "w8" preset does NOT force unnormalized "fp8_qkpv"
            self.assertNotIn("fp8_qkpv", popts)

        with patch.dict(
            os.environ,
            {
                "SGLANG_TPU_KV_PACK_DENSE2": "1",
                "SGLANG_TPU_RPA_V3_DECODE_FAST": "w7",
                "SGLANG_TPU_RPA_V3_DECODE_FAST_W8": "1",
            },
            clear=False,
        ):
            _invoke_v3_kernel(
                _fake_rpa,
                queries=jax.ShapeDtypeStruct((1, 80, 8, 2, 128), jnp.bfloat16),
                keys=jax.ShapeDtypeStruct((80, 1, 4, 128), jnp.float8_e4m3fn),
                values=jax.ShapeDtypeStruct((80, 1024), jnp.bfloat16),
                kv_cache_fused=c_d2,
                kv_lens=jax.ShapeDtypeStruct((16,), jnp.int32),
                page_indices=jax.ShapeDtypeStruct((16 * 17,), jnp.int32),
                cu_q_lens=jax.ShapeDtypeStruct((17,), jnp.int32),
                cu_kv_lens=jax.ShapeDtypeStruct((17,), jnp.int32),
                distribution=dist,
                attention_sink=None,
                sm_scale=0.125,
                sliding_window=2048,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
                decode_query_size=5,
                fused_out_pack=2,
                prepared_inputs=True,
                skip_mixed=True,
                decode_out_norm_eps=1e-5,
                decode_gate_in_values=True,
            )
            self.assertTrue(captured.get("_kv_pack_dense2"))
            self.assertIn("w8", captured.get("_decode_fast", ""))

    def test_kv_pool_dense2_pricing_and_draft_exclusion(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed in lightweight CPU venv")

        from sglang_tpu.dflash_draft_kv_write import (
            _scatter_layer_kv,
            dflash_write_draft_kv,
        )
        from sglang_tpu.hooks.draft_kv_pool import (
            draft_cell_size_per_token,
            draft_mha_pool_class,
        )
        from sglang_tpu.kv_pool import (
            TpuDraftFusedMHATokenToKVPool,
            TpuFusedMHATokenToKVPool,
            kv_pack_dense2_enabled,
            kv_pack_heads,
        )

        with patch.dict(
            os.environ,
            {"SGLANG_TPU_KV_PACK_HEADS": "1", "SGLANG_TPU_KV_PACK_DENSE2": "1"},
            clear=False,
        ):
            self.assertEqual(kv_pack_heads(2, 64, 128), 2)
            self.assertTrue(
                kv_pack_dense2_enabled(
                    packs_heads_cls=True,
                    num_kv_heads=1,
                    head_dim=128,
                    align=128,
                    store_dtype=torch.float8_e4m3fn,
                )
            )
            # DFlash draft pool (num_kv_heads=1, head_dim=64) MUST be excluded!
            self.assertFalse(
                kv_pack_dense2_enabled(
                    packs_heads_cls=True,
                    num_kv_heads=1,
                    head_dim=64,
                    align=128,
                    store_dtype=torch.float8_e4m3fn,
                )
            )
            self.assertFalse(
                kv_pack_dense2_enabled(
                    packs_heads_cls=True,
                    num_kv_heads=1,
                    head_dim=128,
                    align=128,
                    store_dtype=torch.bfloat16,
                )
            )

            with patch.dict(os.environ, {"SGLANG_TPU_KV_PACK_DENSE2": "0"}, clear=False):
                b_stock = TpuFusedMHATokenToKVPool.kv_bytes_per_token_per_layer(
                    num_kv_heads=2,
                    head_dim=64,
                    v_head_dim=64,
                    store_dtype=torch.float8_e4m3fn,
                )
            b_dense2 = TpuFusedMHATokenToKVPool.kv_bytes_per_token_per_layer(
                num_kv_heads=2,
                head_dim=64,
                v_head_dim=64,
                store_dtype=torch.float8_e4m3fn,
            )
            self.assertEqual(b_stock, 512)
            self.assertEqual(b_dense2, 256)

            pool = TpuFusedMHATokenToKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                head_num=2,
                head_dim=64,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
            )
            self.assertTrue(pool.kv_pack_heads)
            self.assertTrue(pool.kv_pack_dense2)
            self.assertTrue(pool.kv_layout_tag().endswith("-d2"))
            self.assertEqual(tuple(pool.kv_buffer[0].shape), (3, 64, 1, 4, 128))
            self.assertEqual(tuple(pool._logical_kv_view(0).shape), (3 * 128, 2, 128))

            # Muse Spark target passes pre-folded (head_num=1, head_dim=128) -> dense2 enabled,
            # while DFlash draft also has (head_num=1, head_dim=128) at TP=8 and MUST stay on
            # 1-tok/row TpuDraftFusedMHATokenToKVPool because draft writers emit 1 tok/row (#1539).
            target_1x128 = TpuFusedMHATokenToKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                head_num=1,
                head_dim=128,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
            )
            self.assertEqual(target_1x128.kv_pack_heads, 1)
            self.assertTrue(target_1x128.kv_pack_dense2)
            self.assertTrue(target_1x128.kv_layout_tag().endswith("-d2"))
            self.assertEqual(tuple(target_1x128.kv_buffer[0].shape), (3, 64, 1, 4, 128))

            draft_1x128 = TpuDraftFusedMHATokenToKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                head_num=1,
                head_dim=128,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
            )
            self.assertIsInstance(draft_1x128, TpuFusedMHATokenToKVPool)
            self.assertFalse(draft_1x128.kv_pack_dense2)
            self.assertFalse(draft_1x128.kv_layout_tag().endswith("-d2"))
            self.assertEqual(tuple(draft_1x128.kv_buffer[0].shape), (3, 128, 1, 4, 128))
            self.assertEqual(
                TpuDraftFusedMHATokenToKVPool.kv_bytes_per_token_per_layer(
                    num_kv_heads=1,
                    head_dim=128,
                    v_head_dim=128,
                    store_dtype=torch.float8_e4m3fn,
                ),
                512,
            )

            import types

            target_kvc = types.SimpleNamespace(
                is_draft_worker=False,
                speculative_draft_attention_backend=None,
            )
            draft_kvc_none = types.SimpleNamespace(
                is_draft_worker=True,
                speculative_draft_attention_backend=None,
            )
            draft_kvc_rpa = types.SimpleNamespace(
                is_draft_worker=True,
                speculative_draft_attention_backend="pallas_tpu_rpa_v3",
            )
            self.assertIs(
                draft_mha_pool_class(lambda _k: TpuFusedMHATokenToKVPool, target_kvc),
                TpuFusedMHATokenToKVPool,
            )
            self.assertIs(
                draft_mha_pool_class(lambda _k: TpuFusedMHATokenToKVPool, draft_kvc_none),
                TpuDraftFusedMHATokenToKVPool,
            )
            with patch(
                "sglang_tpu.tpu_platform._mha_kv_pool_cls",
                return_value=TpuFusedMHATokenToKVPool,
            ):
                self.assertIs(
                    draft_mha_pool_class(
                        lambda _k: TpuFusedMHATokenToKVPool, draft_kvc_rpa
                    ),
                    TpuDraftFusedMHATokenToKVPool,
                )
                d_mc = types.SimpleNamespace(
                    get_num_kv_heads=lambda _tp: 1,
                    head_dim=128,
                    v_head_dim=128,
                )
                with patch(
                    "sglang.srt.runtime_context.get_spec",
                    return_value=types.SimpleNamespace(
                        speculative_draft_attention_backend="pallas_tpu_rpa_v3"
                    ),
                ):
                    self.assertEqual(
                        draft_cell_size_per_token(
                            lambda **_kw: 1 * 256 * 5 * 1,
                            draft_model_config=d_mc,
                            draft_num_layers=5,
                            draft_kv_cache_dtype=torch.float8_e4m3fn,
                            tp_size=8,
                        ),
                        5 * 512,
                    )

            with self.assertRaisesRegex(NotImplementedError, "non-dense2"):
                dflash_write_draft_kv(
                    draft_model=types.SimpleNamespace(),
                    kv_pool=target_1x128,
                    ctx_hidden=torch.zeros((1, 8)),
                    ctx_positions=torch.zeros((1,), dtype=torch.int32),
                    ctx_cache_loc=torch.zeros((1,), dtype=torch.int32),
                    cache_loc_2d=None,
                    commit_lens=None,
                    max_num_reqs=1,
                )
            with self.assertRaisesRegex(NotImplementedError, "non-dense2"):
                _scatter_layer_kv(
                    kv_pool=target_1x128,
                    layer_idx=0,
                    loc=torch.zeros((1,), dtype=torch.int32),
                    k=torch.zeros((1, 1, 128), dtype=torch.float8_e4m3fn),
                    v=torch.zeros((1, 1, 128), dtype=torch.float8_e4m3fn),
                )
            self.assertEqual(target_1x128.kv_buffer[0].nbytes, 3 * 128 * 256)

            draft_1x64 = TpuFusedMHATokenToKVPool(
                size=256,
                page_size=128,
                dtype=torch.float8_e4m3fn,
                head_num=1,
                head_dim=64,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
            )
            self.assertEqual(draft_1x64.kv_pack_heads, 1)
            self.assertFalse(draft_1x64.kv_pack_dense2)
            self.assertFalse(draft_1x64.kv_layout_tag().endswith("-d2"))
            self.assertEqual(tuple(draft_1x64.kv_buffer[0].shape), (3, 128, 1, 4, 128))
            self.assertEqual(draft_1x64.kv_buffer[0].nbytes, 3 * 128 * 512)

    def test_rpad_fast_dense2_and_w8_cpu_interpret_exactness(self):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        rng = np.random.default_rng(20261121)
        b_sz = 16
        pps = 17
        n_pages = b_sz * pps + 4
        sm_scale = 0.125

        kv_lens = np.array(
            [1, 2, 127, 128, 129, 255, 256, 301, 512, 513, 777, 1024, 1025, 1400, 64, 190],
            dtype=np.int32,
        )
        page_indices = np.arange(b_sz * pps, dtype=np.int32)
        cu_q_lens = np.arange(b_sz + 1, dtype=np.int32)
        cu_kv_lens = np.arange(b_sz + 1, dtype=np.int32) * (pps * 128)
        dist = np.array([14, 14, 14], dtype=np.int32)

        stock_u8 = np.full((n_pages, 128, 1, 4, 128), 0x7F, dtype=np.uint8)
        d2_u8 = np.full((n_pages, 64, 1, 4, 128), 0x7F, dtype=np.uint8)
        valid_tok_u8 = _rand_fp8_u8(rng, (n_pages, 128, 2, 128), scale=0.4)

        for s in range(14):
            kvl = int(kv_lens[s])
            for pos in range(kvl - 1):
                pg = int(page_indices[s * pps + pos // 128])
                off = pos % 128
                stock_u8[pg, off, 0, 0, :] = valid_tok_u8[pg, off, 0, :]
                stock_u8[pg, off, 0, 1, :] = valid_tok_u8[pg, off, 1, :]
                stock_u8[pg, off, 0, 2:, :] = 0
                row = off // 2
                if (off & 1) == 0:
                    d2_u8[pg, row, 0, 0, :] = valid_tok_u8[pg, off, 0, :]
                    d2_u8[pg, row, 0, 1, :] = valid_tok_u8[pg, off, 1, :]
                else:
                    d2_u8[pg, row, 0, 2, :] = valid_tok_u8[pg, off, 0, :]
                    d2_u8[pg, row, 0, 3, :] = valid_tok_u8[pg, off, 1, :]

        q_3d = _rand_muse_q_3d(rng, b_sz, scale=0.35)
        gate_2d = (rng.standard_normal((b_sz, 1024)) * 0.5).astype(np.float32)
        kv_new_u8 = np.zeros((b_sz, 1, 4, 128), dtype=np.uint8)
        kv_new_u8[:, 0, :2, :] = _rand_fp8_u8(rng, (b_sz, 2, 128), scale=0.4)

        q_5d = _pack_q_5d(q_3d)
        kv_new_f8 = _u8_to_fp8(kv_new_u8)
        gate_jnp = jnp.asarray(gate_2d, dtype=jnp.bfloat16)

        for tag, mode_str, sw in (
            ("swa_w7_d2", "w7", 2048),
            ("swa_w8_d2", "w8", 2048),
            ("swa_w8_adapt_d2", "w8,c1024,adapt", 2048),
            ("full_w7_d2", "w7", None),
            ("full_w8_d2", "w8", None),
            ("full_w8_adapt_d2", "w8,c1024,adapt", None),
        ):
            with self.subTest(case=tag):
                ref_out, _ = _ref_muse_decode_or_verify(
                    q_3d,
                    kv_new_u8,
                    stock_u8,
                    kv_lens,
                    page_indices,
                    num_active=14,
                    dq=1,
                    pps=pps,
                    sm_scale=sm_scale,
                    sliding_window=sw,
                    gate_bf16=gate_2d,
                    out_norm_eps=1e-5,
                )
                out_st, _ = run_decode_fast_rpa(
                    q_5d,
                    kv_new_f8,
                    None,
                    _u8_to_fp8(stock_u8),
                    jnp.asarray(kv_lens),
                    jnp.asarray(page_indices),
                    jnp.asarray(cu_q_lens),
                    jnp.asarray(cu_kv_lens),
                    jnp.asarray(dist),
                    sm_scale=sm_scale,
                    sliding_window=sw,
                    k_scale=1.0,
                    v_scale=1.0,
                    mode_str=mode_str,
                    fused_out_pack=2,
                    prepared_inputs=True,
                    gate=gate_jnp,
                    out_norm_eps=1e-5,
                    interpret=True,
                    kv_pack_dense2=False,
                )
                out_d2, upd_d2 = run_decode_fast_rpa(
                    q_5d,
                    kv_new_f8,
                    None,
                    _u8_to_fp8(d2_u8),
                    jnp.asarray(kv_lens),
                    jnp.asarray(page_indices),
                    jnp.asarray(cu_q_lens),
                    jnp.asarray(cu_kv_lens),
                    jnp.asarray(dist),
                    sm_scale=sm_scale,
                    sliding_window=sw,
                    k_scale=1.0,
                    v_scale=1.0,
                    mode_str=mode_str,
                    fused_out_pack=2,
                    prepared_inputs=True,
                    gate=gate_jnp,
                    out_norm_eps=1e-5,
                    interpret=True,
                    kv_pack_dense2=True,
                )
                out_st_np = np.asarray(out_st.astype(jnp.float32))
                out_d2_np = np.asarray(out_d2.astype(jnp.float32))
                self.assertFalse(np.isnan(out_d2_np[:14]).any(), f"NaN in {tag}")
                np.testing.assert_allclose(out_d2_np[:14], out_st_np[:14], rtol=5e-3, atol=5e-3)
                np.testing.assert_allclose(out_d2_np[:14], ref_out[:14], rtol=1.5e-2, atol=1.5e-2)

                # Verify written pair-row (including preserved even partner when pos is odd!)
                upd_d2_u8 = _fp8_to_u8(upd_d2)
                for s in range(14):
                    pos = int(kv_lens[s]) - 1
                    pg = int(page_indices[s * pps + pos // 128])
                    off = pos % 128
                    got_k, got_v = _unpack_dense2_u8_token(upd_d2_u8, pg, off)
                    np.testing.assert_array_equal(got_k, kv_new_u8[s, 0, 0, :])
                    np.testing.assert_array_equal(got_v, kv_new_u8[s, 0, 1, :])
                    if (off & 1) == 1:
                        prev_k, prev_v = _unpack_dense2_u8_token(upd_d2_u8, pg, off - 1)
                        np.testing.assert_array_equal(prev_k, valid_tok_u8[pg, off - 1, 0, :])
                        np.testing.assert_array_equal(prev_v, valid_tok_u8[pg, off - 1, 1, :])

    def test_rpad_fast_dense2_multistep_page_crossing_interpret(self):
        """3 consecutive decode steps crossing `127 -> 128 -> 129` on `kv_pack_dense2=True`."""
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        rng = np.random.default_rng(77)
        b_sz = 4
        pps = 4
        n_pages = b_sz * pps + 2
        base_lens = np.array([126, 127, 1, 255], dtype=np.int32)
        page_indices = np.arange(b_sz * pps, dtype=np.int32)
        cu_q_lens = np.arange(b_sz + 1, dtype=np.int32)
        cu_kv_lens = np.arange(b_sz + 1, dtype=np.int32) * (pps * 128)
        dist = np.array([b_sz, b_sz, b_sz], dtype=np.int32)

        stock_u8 = np.full((n_pages, 128, 1, 4, 128), 0x7F, dtype=np.uint8)
        init_tok = _rand_fp8_u8(rng, (n_pages, 128, 2, 128), scale=0.4)
        for s in range(b_sz):
            for pos in range(int(base_lens[s])):
                pg = int(page_indices[s * pps + pos // 128])
                off = pos % 128
                stock_u8[pg, off, 0, 0, :] = init_tok[pg, off, 0, :]
                stock_u8[pg, off, 0, 1, :] = init_tok[pg, off, 1, :]
                stock_u8[pg, off, 0, 2:, :] = 0
        cur_d2_u8 = _pack_stock_cache_to_dense2_u8(stock_u8)
        cur_stock_u8 = stock_u8.copy()

        for step in range(3):
            kv_lens = base_lens + (step + 1)
            q_3d = _rand_muse_q_3d(rng, b_sz, scale=0.3)
            kv_new_u8 = np.zeros((b_sz, 1, 4, 128), dtype=np.uint8)
            kv_new_u8[:, 0, :2, :] = _rand_fp8_u8(rng, (b_sz, 2, 128), scale=0.4)

            ref_out, cur_stock_u8 = _ref_muse_decode_or_verify(
                q_3d,
                kv_new_u8,
                cur_stock_u8,
                kv_lens,
                page_indices,
                num_active=b_sz,
                dq=1,
                pps=pps,
                sm_scale=0.125,
                sliding_window=2048,
            )
            out_d2, upd_d2 = run_decode_fast_rpa(
                _pack_q_5d(q_3d),
                _u8_to_fp8(kv_new_u8),
                None,
                _u8_to_fp8(cur_d2_u8),
                jnp.asarray(kv_lens),
                jnp.asarray(page_indices),
                jnp.asarray(cu_q_lens),
                jnp.asarray(cu_kv_lens),
                jnp.asarray(dist),
                sm_scale=0.125,
                sliding_window=2048,
                k_scale=1.0,
                v_scale=1.0,
                mode_str="w8",
                fused_out_pack=2,
                prepared_inputs=True,
                interpret=True,
                kv_pack_dense2=True,
            )
            cur_d2_u8 = _merge_interpret_aliased_d2_u8(
                cur_d2_u8,
                _fp8_to_u8(upd_d2),
                kv_lens,
                page_indices,
                num_active=b_sz,
                dq=1,
                pps=pps,
            )
            np.testing.assert_allclose(
                np.asarray(out_d2.astype(jnp.float32)), ref_out, rtol=1.5e-2, atol=1.5e-2
            )

    def test_dflash_verify_dq4_dq5_dq6_fused_gate_norm_and_dense2_interpret(self):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            can_use_decode_fast,
            run_decode_fast_rpa,
        )

        rng = np.random.default_rng(805)
        b_sz = 4
        pps = 6
        n_pages = b_sz * pps + 2

        for dq in (4, 5, 6):
            total_tok = b_sz * dq
            q_dummy = jax.ShapeDtypeStruct((total_tok, 16, 128), jnp.bfloat16)
            c_stock_shape = jax.ShapeDtypeStruct((n_pages, 128, 1, 4, 128), jnp.float8_e4m3fn)
            c_d2_shape = jax.ShapeDtypeStruct((n_pages, 64, 1, 4, 128), jnp.float8_e4m3fn)
            self.assertTrue(
                can_use_decode_fast(
                    mode_str="w8",
                    q=q_dummy,
                    kv_cache_fused=c_stock_shape,
                    out_dtype=jnp.bfloat16,
                    decode_query_size=dq,
                    custom_mask=None,
                    attention_sink=None,
                    soft_cap=None,
                    xai_temperature_len=None,
                    q_scale=None,
                    actual_num_q_heads=16,
                    actual_num_kv_heads=1,
                    actual_head_dim=128,
                    sliding_window=2048,
                    pages_per_seq=pps,
                    kv_pack_dense2=False,
                    need_fused_post=True,
                )
            )
            self.assertTrue(
                can_use_decode_fast(
                    mode_str="w8",
                    q=q_dummy,
                    kv_cache_fused=c_d2_shape,
                    out_dtype=jnp.bfloat16,
                    decode_query_size=dq,
                    custom_mask=None,
                    attention_sink=None,
                    soft_cap=None,
                    xai_temperature_len=None,
                    q_scale=None,
                    actual_num_q_heads=16,
                    actual_num_kv_heads=1,
                    actual_head_dim=128,
                    sliding_window=None,
                    pages_per_seq=pps,
                    kv_pack_dense2=True,
                    need_fused_post=True,
                )
            )

            kv_lens = np.array([dq, 127 + dq, 260, 100], dtype=np.int32)
            page_indices = np.arange(b_sz * pps, dtype=np.int32)
            cu_q_lens = np.arange(b_sz + 1, dtype=np.int32) * dq
            cu_kv_lens = np.arange(b_sz + 1, dtype=np.int32) * (pps * 128)
            dist = np.array([3, 3, 3], dtype=np.int32)

            stock_u8 = np.full((n_pages, 128, 1, 4, 128), 0x7F, dtype=np.uint8)
            d2_u8 = np.full((n_pages, 64, 1, 4, 128), 0x7F, dtype=np.uint8)
            pref_tok = _rand_fp8_u8(rng, (n_pages, 128, 2, 128), scale=0.4)
            for s in range(3):
                for pos in range(int(kv_lens[s]) - dq):
                    pg = int(page_indices[s * pps + pos // 128])
                    off = pos % 128
                    stock_u8[pg, off, 0, 0, :] = pref_tok[pg, off, 0, :]
                    stock_u8[pg, off, 0, 1, :] = pref_tok[pg, off, 1, :]
                    stock_u8[pg, off, 0, 2:, :] = 0
                    row = off // 2
                    if (off & 1) == 0:
                        d2_u8[pg, row, 0, 0, :] = pref_tok[pg, off, 0, :]
                        d2_u8[pg, row, 0, 1, :] = pref_tok[pg, off, 1, :]
                    else:
                        d2_u8[pg, row, 0, 2, :] = pref_tok[pg, off, 0, :]
                        d2_u8[pg, row, 0, 3, :] = pref_tok[pg, off, 1, :]

            q_3d = _rand_muse_q_3d(rng, total_tok, scale=0.3)
            gate_2d = (rng.standard_normal((total_tok, 1024)) * 0.5).astype(np.float32)
            kv_new_u8 = np.zeros((total_tok, 1, 4, 128), dtype=np.uint8)
            kv_new_u8[:, 0, :2, :] = _rand_fp8_u8(rng, (total_tok, 2, 128), scale=0.4)

            for sw in (2048, None):
                with self.subTest(dq=dq, sw=sw):
                    ref_out, ref_stock = _ref_muse_decode_or_verify(
                        q_3d,
                        kv_new_u8,
                        stock_u8,
                        kv_lens,
                        page_indices,
                        num_active=3,
                        dq=dq,
                        pps=pps,
                        sm_scale=0.125,
                        sliding_window=sw,
                        gate_bf16=gate_2d,
                        out_norm_eps=1e-5,
                    )
                    outs = []
                    for use_d2, init_c in ((False, stock_u8), (True, d2_u8)):
                        out_k, upd_k = run_decode_fast_rpa(
                            _pack_q_5d(q_3d),
                            _u8_to_fp8(kv_new_u8),
                            None,
                            _u8_to_fp8(init_c),
                            jnp.asarray(kv_lens),
                            jnp.asarray(page_indices),
                            jnp.asarray(cu_q_lens),
                            jnp.asarray(cu_kv_lens),
                            jnp.asarray(dist),
                            sm_scale=0.125,
                            sliding_window=sw,
                            k_scale=1.0,
                            v_scale=1.0,
                            mode_str="w8",
                            decode_query_size=dq,
                            fused_out_pack=2,
                            prepared_inputs=True,
                            gate=jnp.asarray(gate_2d, dtype=jnp.bfloat16),
                            out_norm_eps=1e-5,
                            interpret=True,
                            kv_pack_dense2=use_d2,
                        )
                        got = np.asarray(out_k.astype(jnp.float32))
                        outs.append(got)
                        self.assertFalse(np.isnan(got[: 3 * dq]).any())
                        np.testing.assert_allclose(
                            got[: 3 * dq], ref_out[: 3 * dq], rtol=1.5e-2, atol=1.5e-2
                        )
                        if use_d2:
                            merged_u8 = _merge_interpret_aliased_d2_u8(
                                d2_u8,
                                _fp8_to_u8(upd_k),
                                kv_lens,
                                page_indices,
                                num_active=3,
                                dq=dq,
                                pps=pps,
                            )
                            for s in range(3):
                                for pos in range(int(kv_lens[s])):
                                    pg = int(page_indices[s * pps + pos // 128])
                                    off = pos % 128
                                    gk, gv = _unpack_dense2_u8_token(merged_u8, pg, off)
                                    np.testing.assert_array_equal(gk, ref_stock[pg, off, 0, 0, :])
                                    np.testing.assert_array_equal(gv, ref_stock[pg, off, 0, 1, :])
                    # Verify stock and dense2 verify outputs match identically!
                    np.testing.assert_allclose(outs[0][: 3 * dq], outs[1][: 3 * dq], rtol=5e-3, atol=5e-3)

    def test_write_kv_only_dense2_cpu_interpret_odd_boundaries(self):
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import write_kv_only_rpa_v3

        rng = np.random.default_rng(909)
        b_sz = 4
        pps = 4
        n_pages = b_sz * pps + 2
        t_total = 512

        pref_lens = np.array([129, 65, 0, 0], dtype=np.int32)
        q_lens = np.array([131, 7, 199, 0], dtype=np.int32)
        kv_lens = pref_lens + q_lens
        cu_q_lens = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        cu_kv_lens = np.arange(b_sz + 1, dtype=np.int32) * (pps * 128)
        page_indices = np.arange(b_sz * pps, dtype=np.int32)
        dist = np.array([0, 0, 3], dtype=np.int32)

        d2_u8 = np.full((n_pages, 64, 1, 4, 128), 0x7F, dtype=np.uint8)
        stock_ref = np.zeros((n_pages, 128, 2, 128), dtype=np.uint8)
        init_tok = rng.integers(0, 0x70, size=(n_pages, 128, 2, 128), dtype=np.uint8)

        for s in range(3):
            for pos in range(int(pref_lens[s])):
                pg = int(page_indices[s * pps + pos // 128])
                off = pos % 128
                stock_ref[pg, off] = init_tok[pg, off]
                row = off // 2
                if (off & 1) == 0:
                    d2_u8[pg, row, 0, 0, :] = init_tok[pg, off, 0, :]
                    d2_u8[pg, row, 0, 1, :] = init_tok[pg, off, 1, :]
                else:
                    d2_u8[pg, row, 0, 2, :] = init_tok[pg, off, 0, :]
                    d2_u8[pg, row, 0, 3, :] = init_tok[pg, off, 1, :]

        kv_new_u8 = np.zeros((t_total, 1, 4, 128), dtype=np.uint8)
        kv_new_u8[:, 0, :2, :] = rng.integers(0, 0x70, size=(t_total, 2, 128), dtype=np.uint8)
        for s in range(3):
            qs = int(cu_q_lens[s])
            ql = int(q_lens[s])
            kv_st = int(pref_lens[s])
            for j in range(ql):
                pos = kv_st + j
                pg = int(page_indices[s * pps + pos // 128])
                stock_ref[pg, pos % 128, 0, :] = kv_new_u8[qs + j, 0, 0, :]
                stock_ref[pg, pos % 128, 1, :] = kv_new_u8[qs + j, 0, 1, :]

        upd_d2 = write_kv_only_rpa_v3(
            _u8_to_fp8(kv_new_u8),
            _u8_to_fp8(d2_u8),
            jnp.asarray(kv_lens),
            jnp.asarray(page_indices),
            jnp.asarray(cu_q_lens),
            jnp.asarray(cu_kv_lens),
            jnp.asarray(dist),
            interpret=True,
            kv_pack_dense2=True,
        )
        self.assertEqual(upd_d2.shape, (n_pages, 64, 1, 4, 128))
        raw_upd_u8 = _fp8_to_u8(upd_d2)

        # 1. Directly check raw `upd_d2` on all written pair-rows `[pair_lo, pair_hi)`
        #    (including preserved even partner at `kv_start - 1` when `kv_start` is odd!),
        #    and verify untouched poison (`0x7F`) or prefix rows were never corrupted.
        written_mask = np.zeros((n_pages, 64), dtype=bool)
        for s in range(3):
            kv_st = int(pref_lens[s])
            kvl = int(kv_lens[s])
            for p in range(kv_st // 128, (kvl + 127) // 128):
                t_lo = max(kv_st, p * 128) - p * 128
                t_hi = min(kvl, (p + 1) * 128) - p * 128
                r_lo = t_lo // 2
                r_hi = (t_hi + 1) // 2
                pg = int(page_indices[s * pps + p])
                written_mask[pg, r_lo:r_hi] = True
                for off in range(2 * r_lo, min(kvl - p * 128, 2 * r_hi)):
                    gk, gv = _unpack_dense2_u8_token(raw_upd_u8, pg, off)
                    np.testing.assert_array_equal(gk, stock_ref[pg, off, 0, :])
                    np.testing.assert_array_equal(gv, stock_ref[pg, off, 1, :])
        untouched = (~written_mask)[:, :, None, None, None]
        self.assertTrue(np.all(raw_upd_u8[(d2_u8 == 0x7F) & untouched] == 0x7F))
        self.assertTrue(
            np.all(((raw_upd_u8 == d2_u8) | (raw_upd_u8 == 0x7F))[np.broadcast_to(untouched, d2_u8.shape)])
        )

    def test_tpu7x_mosaic_exports_dense2_verify_and_prefill_w8(self):
        try:
            from jax._src import tpu_info as _tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available in lightweight CPU venv")

        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
            write_kv_only_rpa_v3,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            # 1. RPAd_fast w8 + dense2 (d=1) and DFlash verify (dq=5, dq=6) with fused gate+onorm
            for dtag, dq, b_seqs, sw, use_d2, mode_arg, expected_sub in (
                ("RPAd_w8_swa_d2", 1, 64, 2048, True, "w8", "swabatch-mp17-d2"),
                ("RPAd_w8_full_d2", 1, 64, None, True, "w8", "w7auto-mp160"),
                ("RPAd_w8_adapt_swa_d2", 1, 64, 2048, True, "w8,c1024,adapt", "swabatch-mp8-d2"),
                ("RPAd_w8_adapt_full_d2", 1, 64, None, True, "w8,c1024,adapt", "stream0b-mp8-d2"),
                ("RPAd_verify_dq5_swa_gate", 5, 64, 2048, False, "w8", "dq5"),
                ("RPAd_verify_dq5_full_d2_gate", 5, 64, None, True, "w8", "dq5"),
                ("RPAd_verify_dq6_swa_d2_gate", 6, 64, 2048, True, "w8", "dq6"),
            ):
                with self.subTest(export=dtag):
                    phys_pg = 64 if use_d2 else 128
                    tot_tok = b_seqs * dq
                    pps = 129
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q, kv, g, c, kl, pi, cq, ckv, d, _dq=dq, _sw=sw, _d2=use_d2, _df=mode_arg: ragged_paged_attention(
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
                                sliding_window=_sw,
                                k_scale=1.0,
                                v_scale=1.0,
                                decode_query_size=_dq,
                                _fused_out_pack=2,
                                _prepared_inputs=True,
                                _skip_mixed=True,
                                _decode_fast=_df,
                                _decode_out_norm_eps=1e-5,
                                _decode_gate_in_values=True,
                                _kv_pack_dense2=_d2,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, tot_tok, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((tot_tok, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((tot_tok, 1024), jnp.bfloat16),
                        jax.ShapeDtypeStruct((16400, phys_pg, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((b_seqs,), jnp.int32),
                        jax.ShapeDtypeStruct((b_seqs * pps,), jnp.int32),
                        jax.ShapeDtypeStruct((b_seqs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((b_seqs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("RPAd_fast", hlo)
                    self.assertIn(expected_sub, hlo)
                    if use_d2:
                        self.assertIn("-d2", hlo)

            # 2. RPAm Stationary-Q v2: fixed gqa2_vreg, w8 (vreg_acc + sw_short_fuse),
            #    fp8_qkpv_norm + fp8_qkpv_s0, and _kv_pack_dense2=True (SWA & Full)
            base_sw = {
                "stationary_q_m0_f32e_fuse",
                "skip_empty_cdma",
                "mixed_s3",
                "sw2048_fuse",
            }
            base_fl = {
                "stationary_q_m0_f32e_fuse",
                "skip_empty_cdma",
                "mixed_s3",
                "bkv_full_unroll",
            }
            for ptag, psw, mb, opts, use_d2 in (
                (
                    "RPAm_sw_gqa2_vreg_fp8qk",
                    2048,
                    (256, 2048, 256, 256),
                    frozenset(base_sw | {"fp8_qk", "gqa2_vreg"}),
                    False,
                ),
                (
                    "RPAm_sw_w8_vreg_acc_short_fuse",
                    2048,
                    (256, 2048, 256, 256),
                    frozenset(base_sw | {"fp8_qk", "vreg_acc", "sw_short_fuse"}),
                    False,
                ),
                (
                    "RPAm_sw_w8_dense2",
                    2048,
                    (256, 2048, 256, 256),
                    frozenset(
                        base_sw
                        | {"fp8_qkpv", "fp8_qkpv_norm", "fp8_qkpv_s0", "vreg_acc", "sw_short_fuse"}
                    ),
                    True,
                ),
                (
                    "RPAm_full_w8_dense2",
                    None,
                    (512, 2048, 256, 256),
                    frozenset(base_fl | {"fp8_qkpv", "fp8_qkpv_norm"}),
                    True,
                ),
            ):
                with self.subTest(export=ptag):
                    phys_pg = 64 if use_d2 else 128
                    hlo_p = jax_export.export(
                        jax.jit(
                            lambda q, kv, c, kl, pi, cq, ckv, d, _sw=psw, _mb=mb, _op=opts, _d2=use_d2: ragged_paged_attention(
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
                                sliding_window=_sw,
                                k_scale=1.0,
                                v_scale=1.0,
                                m_block_sizes=_mb,
                                _prefill_opts=_op,
                                _fused_out_pack=2,
                                _prepared_inputs=True,
                                _skip_decode=True,
                                _a3_out_layout=True,
                                _kv_pack_dense2=_d2,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, 8192, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((8192, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((1048, phys_pg, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((8,), jnp.int32),
                        jax.ShapeDtypeStruct((8 * 129,), jnp.int32),
                        jax.ShapeDtypeStruct((9,), jnp.int32),
                        jax.ShapeDtypeStruct((9,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("RPAm", hlo_p)
                    if use_d2:
                        self.assertIn("-d2", hlo_p)

            # 3. write_kv_only_rpa_v3 with kv_pack_dense2=True
            hlo_w = jax_export.export(
                jax.jit(
                    lambda kv, c, kl, pi, cq, ckv, d: write_kv_only_rpa_v3(
                        kv, c, kl, pi, cq, ckv, d, kv_pack_dense2=True
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((16384, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((1048, 64, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((8,), jnp.int32),
                jax.ShapeDtypeStruct((8 * 129,), jnp.int32),
                jax.ShapeDtypeStruct((9,), jnp.int32),
                jax.ShapeDtypeStruct((9,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("RPA_write_kv_only_d2", hlo_w)

    def test_write_kv_only_dense2_dynamo_prewarm_and_export(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed in lightweight CPU venv")

        from sglang_tpu.attention_pallas_rpa_v3 import _get_write_kv_only_op

        op_d2 = _get_write_kv_only_op(True)
        self.assertIsNotNone(op_d2)

        def _fn(kv, c, kl, pi, cq, ckv, d):
            op_d2(kv, c, kl, pi, cq, ckv, d, None)
            return kv

        gm, _ = torch._dynamo.export(
            _fn,
            aten_graph=True,
            tracing_mode="fake",
        )(
            torch.zeros((256, 1, 4, 128), dtype=torch.float8_e4m3fn),
            torch.zeros((16, 64, 1, 4, 128), dtype=torch.float8_e4m3fn),
            torch.zeros((4,), dtype=torch.int32),
            torch.zeros((4 * 4,), dtype=torch.int32),
            torch.zeros((5,), dtype=torch.int32),
            torch.zeros((5,), dtype=torch.int32),
            torch.zeros((3,), dtype=torch.int32),
        )
        self.assertIn("rpa_v3_write_kv_only_d2", str(gm.graph))


if __name__ == "__main__":
    unittest.main()
