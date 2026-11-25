# SPDX-License-Identifier: Apache-2.0
"""Unit, CPU interpret-mode (0x7F NaN & race check), and TPU7x Mosaic export tests for Long-Context Decode RPA."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np
import torch


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


def _fp8_from_u8(arr_u8: np.ndarray) -> np.ndarray:
    return np.asarray(_u8_to_fp8(arr_u8), dtype=np.float32)


def _pack_stock_cache_to_dense2_u8(stock_u8: np.ndarray, page_size: int) -> np.ndarray:
    n_pg = stock_u8.shape[0]
    phys = page_size // 2
    d2 = np.full((n_pg, phys, 1, 4, 128), 0x7F, dtype=np.uint8)
    d2[:, :, 0, 0, :] = stock_u8[:, 0::2, 0, 0, :]
    d2[:, :, 0, 1, :] = stock_u8[:, 0::2, 0, 1, :]
    d2[:, :, 0, 2, :] = stock_u8[:, 1::2, 0, 0, :]
    d2[:, :, 0, 3, :] = stock_u8[:, 1::2, 0, 1, :]
    return d2


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / denom) if denom > 0 else 1.0


def _exact_decode_ref(
    q_bf16: np.ndarray,
    k_new_u8: np.ndarray,
    v_new_u8: np.ndarray,
    cache_u8: np.ndarray,
    kv_lens: np.ndarray,
    page_indices: np.ndarray,
    n_active: int,
    page_size: int,
    sm_scale: float,
    pps: int,
) -> np.ndarray:
    out = np.zeros_like(q_bf16, dtype=np.float32)
    for s in range(n_active):
        kvl = int(kv_lens[s])
        if kvl <= 0:
            continue
        k_list = []
        v_list = []
        for pos in range(kvl - 1):
            pg = int(page_indices[s * pps + (pos // page_size)])
            off = pos % page_size
            k_list.append(_fp8_from_u8(cache_u8[pg, off, 0, 0, :]))
            v_list.append(_fp8_from_u8(cache_u8[pg, off, 0, 1, :]))
        k_list.append(_fp8_from_u8(k_new_u8[s]))
        v_list.append(_fp8_from_u8(v_new_u8[s]))
        k_mat = np.stack(k_list, axis=0)
        v_mat = np.stack(v_list, axis=0)
        q_mat = q_bf16[s].astype(np.float32)
        logits = (q_mat @ k_mat.T) * sm_scale
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= np.sum(probs, axis=1, keepdims=True)
        out[s] = probs @ v_mat
    return out


class TestMuseLongContextDecodeRpa(unittest.TestCase):
    def test_long_context_routing_and_page_size_256(self):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            can_use_decode_fast,
            get_shared_group_size,
            parse_decode_fast_config,
        )

        q_3d = jnp.zeros((64, 16, 128), dtype=jnp.bfloat16)
        c_128 = jnp.zeros((16, 128, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        c_d2_256 = jnp.zeros((16, 128, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        c_d2_128 = jnp.zeros((16, 64, 1, 4, 128), dtype=jnp.float8_e4m3fn)

        # w8 unlocks 64k/128k/256k (pps up to 4096) via ring4k instead of slow fallback
        for pps in (640, 1152, 2177):
            for mode in ("w8,c1024", "w8,c1024,ring4k", "w8,c1024,ring8k", "w8,c1024,shared64"):
                self.assertTrue(
                    can_use_decode_fast(
                        mode_str=mode,
                        q=q_3d,
                        kv_cache_fused=c_d2_128,
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
                        pages_per_seq=pps,
                        kv_pack_dense2=True,
                    ),
                    f"Expected True for mode={mode}, pps={pps}",
                )

        self.assertEqual(get_shared_group_size("shared16_host"), 16)
        self.assertEqual(get_shared_group_size("w8,c1024,shared32"), 32)
        self.assertEqual(get_shared_group_size("w8,c1024,shared64,pbf16"), 64)

        cfg_ring8k = parse_decode_fast_config(
            "w8,c1024,ring8k", sliding_window=None, pages_per_seq=2177, page_size=128
        )
        self.assertEqual(cfg_ring8k.loop_kind, "ring4k")
        self.assertEqual(cfg_ring8k.chunk_sz, 8192)
        self.assertEqual(cfg_ring8k.num_slots, 3)
        self.assertTrue(cfg_ring8k.coalesce_dma)

        cfg_sh64 = parse_decode_fast_config(
            "w8,c1024,shared64,pbf16",
            sliding_window=None,
            pages_per_seq=1089,
            page_size=256,
            shared_group=64,
            kv_pack_dense2=True,
        )
        self.assertEqual(cfg_sh64.loop_kind, "shared64")
        self.assertEqual(cfg_sh64.shared_group_size, 64)
        self.assertEqual(cfg_sh64.chunk_sz, 2048)
        self.assertEqual(cfg_sh64.max_ring_pages, 8)
        self.assertEqual(cfg_sh64.shared_q_tile, 256)
        self.assertTrue(cfg_sh64.pv_bf16)
        self.assertTrue(cfg_sh64.coalesce_dma)
        self.assertFalse(cfg_sh64.hoist_new_kv)

        cfg_sh64_4k = parse_decode_fast_config(
            "w8,c4096,shared64,q512",
            sliding_window=None,
            pages_per_seq=1089,
            page_size=256,
            shared_group=64,
            kv_pack_dense2=True,
        )
        self.assertEqual(cfg_sh64_4k.chunk_sz, 4096)
        self.assertEqual(cfg_sh64_4k.max_ring_pages, 16)
        self.assertEqual(cfg_sh64_4k.shared_q_tile, 512)
        del c_128, c_d2_256

    def test_shared_distribution_group64_and_page_size_256(self):
        from sglang_tpu.attention_shared_prefix import (
            compute_shared_mismatch,
            shared_decode_distribution,
        )

        pages = np.broadcast_to(np.arange(1100, dtype=np.int32), (64, 1100)).copy()
        pages[:, 1024:] += np.arange(64, dtype=np.int32)[:, None] * 100
        lengths = np.full((64,), 1024 * 256 + 100, dtype=np.int32)
        mm = compute_shared_mismatch(pages, 64, group_size=64)
        self.assertEqual(mm.tolist(), [1024])
        plan = shared_decode_distribution(
            pages, lengths, 64, 256, group_size=64, cached_mismatch=mm
        )
        self.assertIsNotNone(plan)
        self.assertEqual(plan.tolist(), [64, 64, 64, 1024])

    def test_shared16_and_shared32_and_ring4k_ps256_interpret_parity(self):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            run_decode_fast_rpa,
        )
        from sglang_tpu.attention_shared_prefix import shared_decode_distribution

        rng = np.random.default_rng(7)
        sm_scale = 1.0 / np.sqrt(64.0)

        for tag, bs, n_act, page_size, use_d2, mode, grp in (
            ("sh16_ps128_d2", 16, 15, 128, True, "w8,c1024,shared16,pbf16", 16),
            ("sh32_ps128_d2", 32, 30, 128, True, "w8,c1024,shared32", 32),
            ("sh16_ps256_d2", 16, 16, 256, True, "w8,c1024,shared16", 16),
            ("ring4k_ps256_d2", 16, 14, 256, True, "w8,c1024,ring4k", 0),
        ):
            with self.subTest(case=tag):
                shared_pgs = 2048 // page_size  # 16 at ps=128, 8 at ps=256
                pps = shared_pgs + 4
                n_phys_pages = 256
                stock_cache_u8 = np.full((n_phys_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
                stock_cache_u8[:, :, 0, :2, :] = _rand_fp8_u8(rng, (n_phys_pages, page_size, 2, 128))
                if use_d2:
                    cache_u8 = _pack_stock_cache_to_dense2_u8(stock_cache_u8, page_size)
                else:
                    cache_u8 = stock_cache_u8

                # Shared prefix on first shared_pgs, unique private pages afterwards
                page_indices = np.zeros((bs, pps), dtype=np.int32)
                page_indices[:, :shared_pgs] = np.arange(1, 1 + shared_pgs, dtype=np.int32)
                nxt = 1 + shared_pgs
                for s in range(bs):
                    for p in range(shared_pgs, pps):
                        page_indices[s, p] = nxt
                        nxt += 1

                kv_lens = np.zeros((bs,), dtype=np.int32)
                for s in range(n_act):
                    kv_lens[s] = shared_pgs * page_size + 5 + (s * 23) % (3 * page_size)

                q_np = (rng.standard_normal((bs, 16, 128)) * 0.35).astype(np.float32)
                q_bf16 = jnp.asarray(q_np, dtype=jnp.bfloat16)
                k_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                v_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                k_new = _u8_to_fp8(k_new_u8)[:, None, :]
                v_new = _u8_to_fp8(v_new_u8)[:, None, :]

                cu_q = jnp.arange(bs + 1, dtype=jnp.int32)
                cu_kv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size)
                dist = jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32)
                shared_pages = None
                if grp > 0:
                    # Make sure shared_decode_distribution triggers: need width > 128 in helper
                    pad_w = max(pps, 129)
                    p_pad = np.zeros((bs, pad_w), dtype=np.int32)
                    p_pad[:, :pps] = page_indices
                    plan = shared_decode_distribution(p_pad, kv_lens, n_act, page_size, group_size=grp)
                    self.assertIsNotNone(plan)
                    shared_pages = jnp.asarray(plan[3:], dtype=jnp.int32)

                out_j, _ = run_decode_fast_rpa(
                    q_bf16,
                    k_new,
                    v_new,
                    _u8_to_fp8(cache_u8),
                    jnp.asarray(kv_lens, dtype=jnp.int32),
                    jnp.asarray(page_indices.ravel(), dtype=jnp.int32),
                    cu_q,
                    cu_kv,
                    dist,
                    sm_scale=sm_scale,
                    sliding_window=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    mode_str=mode,
                    decode_query_size=1,
                    fused_out_pack=1,
                    kv_pack_dense2=use_d2,
                    shared_pages=shared_pages,
                    interpret=True,
                )
                ref = _exact_decode_ref(
                    np.asarray(q_bf16),
                    k_new_u8,
                    v_new_u8,
                    stock_cache_u8,
                    kv_lens,
                    page_indices.ravel(),
                    n_act,
                    page_size,
                    sm_scale,
                    pps,
                )
                out_np = np.asarray(out_j, dtype=np.float32)
                self.assertGreater(_cosine(out_np[:n_act], ref[:n_act]), 0.999)
                if n_act < bs:
                    self.assertEqual(np.max(np.abs(out_np[n_act:])), 0.0)

    def test_shared32_dense2_multistep_odd_kv_write_and_readback(self):
        """Verify odd/even kv_lens multi-step KV write + readback under shared32 + dense2."""
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            run_decode_fast_rpa,
        )
        from sglang_tpu.attention_shared_prefix import shared_decode_distribution

        rng = np.random.default_rng(19)
        sm_scale = 1.0 / np.sqrt(64.0)
        bs, n_act, page_size = 32, 20, 128
        shared_pgs = 16
        pps = shared_pgs + 4
        n_phys_pages = 256
        stock_cache_u8 = np.full((n_phys_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8)
        stock_cache_u8[:, :, 0, :2, :] = _rand_fp8_u8(rng, (n_phys_pages, page_size, 2, 128))
        d2_cache_u8 = _pack_stock_cache_to_dense2_u8(stock_cache_u8, page_size)

        page_indices = np.zeros((bs, pps), dtype=np.int32)
        page_indices[:, :shared_pgs] = np.arange(1, 1 + shared_pgs, dtype=np.int32)
        nxt = 1 + shared_pgs
        for s in range(bs):
            for p in range(shared_pgs, pps):
                page_indices[s, p] = nxt
                nxt += 1

        # Include both even (shared*128 + 128 -> odd cache_kv!) and odd kv_lens across page boundaries
        kv_lens = np.zeros((bs,), dtype=np.int32)
        for s in range(n_act):
            kv_lens[s] = shared_pgs * page_size + 127 + (s % 4)

        cu_q = jnp.arange(bs + 1, dtype=jnp.int32)
        cu_kv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size)
        dist = jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32)
        pad_w = max(pps, 129)
        p_pad = np.zeros((bs, pad_w), dtype=np.int32)
        p_pad[:, :pps] = page_indices

        for step in range(3):
            cur_lens = kv_lens + step
            plan = shared_decode_distribution(p_pad, cur_lens, n_act, page_size, group_size=32)
            self.assertIsNotNone(plan)
            shared_pages = jnp.asarray(plan[3:], dtype=jnp.int32)

            q_bf16 = jnp.asarray(rng.standard_normal((bs, 16, 128)) * 0.35, dtype=jnp.bfloat16)
            k_new_u8 = _rand_fp8_u8(rng, (bs, 128))
            v_new_u8 = _rand_fp8_u8(rng, (bs, 128))

            out_j, upd_cache_j = run_decode_fast_rpa(
                q_bf16,
                _u8_to_fp8(k_new_u8)[:, None, :],
                _u8_to_fp8(v_new_u8)[:, None, :],
                _u8_to_fp8(d2_cache_u8),
                jnp.asarray(cur_lens, dtype=jnp.int32),
                jnp.asarray(page_indices.ravel(), dtype=jnp.int32),
                cu_q,
                cu_kv,
                dist,
                sm_scale=sm_scale,
                sliding_window=None,
                k_scale=1.0,
                v_scale=1.0,
                mode_str="w8,c1024,shared32,pbf16",
                decode_query_size=1,
                fused_out_pack=1,
                kv_pack_dense2=True,
                shared_pages=shared_pages,
                interpret=True,
            )
            ref = _exact_decode_ref(
                np.asarray(q_bf16),
                k_new_u8,
                v_new_u8,
                stock_cache_u8,
                cur_lens,
                page_indices.ravel(),
                n_act,
                page_size,
                sm_scale,
                pps,
            )
            self.assertGreater(_cosine(np.asarray(out_j)[:n_act], ref[:n_act]), 0.999)
            # Apply the written row updates onto both stock_cache_u8 and d2_cache_u8 for next step
            upd_u8 = _fp8_to_u8(upd_cache_j)
            for s in range(n_act):
                pos = int(cur_lens[s]) - 1
                pg = int(page_indices[s, pos // page_size])
                off = pos % page_size
                stock_cache_u8[pg, off, 0, 0, :] = k_new_u8[s]
                stock_cache_u8[pg, off, 0, 1, :] = v_new_u8[s]
                d2_cache_u8[pg, off // 2] = upd_u8[pg, off // 2]

    def test_shsplit_shfp8_active_tile_skip_shared8_and_swa_ps256(self):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            run_decode_fast_rpa,
        )
        from sglang_tpu.attention_shared_prefix import shared_decode_distribution

        rng = np.random.default_rng(42)
        sm_scale = 1.0 / np.sqrt(64.0)

        for tag, bs, n_act, page_size, mode, grp, min_cos in (
            ("sh16_shsplit", 16, 14, 256, "w8,c1024,shared16,shsplit,pbf16", 16, 0.999),
            ("sh16_shfp8", 16, 14, 256, "w8,c1024,shared16,shsplit,shfp8", 16, 0.998),
            ("sh16_shfp8pv", 16, 15, 256, "w8,c1024,shared16,shsplit,shfp8pv", 16, 0.9985),
            ("sh32_c8_tile_skip", 32, 8, 256, "w8,c1024,shared32,shsplit,shfp8", 32, 0.998),
            ("sh8_c8_shfp8", 8, 8, 256, "w8,c1024,shared8,shsplit,shfp8", 8, 0.998),
            ("sh16_q128_c8_tile_skip", 16, 8, 256, "w8,c1024,shared16,q128,shsplit,pbf16", 16, 0.999),
            ("sh16_q128_c16", 16, 16, 256, "w8,c1024,shared16,q128,shsplit,pbf16", 16, 0.999),
        ):
            with self.subTest(case=tag):
                shared_pgs = 2048 // page_size  # 8 full shared pages -> full block
                pps = shared_pgs + 4
                n_phys_pages = 256
                stock_cache_u8 = np.full(
                    (n_phys_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8
                )
                stock_cache_u8[:, :, 0, :2, :] = _rand_fp8_u8(
                    rng, (n_phys_pages, page_size, 2, 128)
                )
                cache_u8 = _pack_stock_cache_to_dense2_u8(stock_cache_u8, page_size)

                page_indices = np.zeros((bs, pps), dtype=np.int32)
                page_indices[:, :shared_pgs] = np.arange(
                    1, 1 + shared_pgs, dtype=np.int32
                )
                nxt = 1 + shared_pgs
                for s in range(bs):
                    for p in range(shared_pgs, pps):
                        page_indices[s, p] = nxt
                        nxt += 1

                kv_lens = np.zeros((bs,), dtype=np.int32)
                for s in range(n_act):
                    kv_lens[s] = shared_pgs * page_size + 7 + (s * 31) % (3 * page_size)

                q_bf16 = jnp.asarray(
                    rng.standard_normal((bs, 16, 128)) * 0.35, dtype=jnp.bfloat16
                )
                k_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                v_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                k_new = _u8_to_fp8(k_new_u8)[:, None, :]
                v_new = _u8_to_fp8(v_new_u8)[:, None, :]
                cu_q = jnp.arange(bs + 1, dtype=jnp.int32)
                cu_kv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size)
                dist = jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32)
                shared_pages = jnp.full((bs // grp,), shared_pgs, dtype=jnp.int32)
                if grp >= 16:
                    pad_w = max(pps, 129)
                    p_pad = np.zeros((bs, pad_w), dtype=np.int32)
                    p_pad[:, :pps] = page_indices
                    plan = shared_decode_distribution(
                        p_pad, kv_lens, n_act, page_size, group_size=grp
                    )
                    self.assertIsNotNone(plan)
                    shared_pages = jnp.asarray(plan[3:], dtype=jnp.int32)

                out_j, _ = run_decode_fast_rpa(
                    q_bf16,
                    k_new,
                    v_new,
                    _u8_to_fp8(cache_u8),
                    jnp.asarray(kv_lens, dtype=jnp.int32),
                    jnp.asarray(page_indices.ravel(), dtype=jnp.int32),
                    cu_q,
                    cu_kv,
                    dist,
                    sm_scale=sm_scale,
                    sliding_window=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    mode_str=mode,
                    decode_query_size=1,
                    fused_out_pack=1,
                    kv_pack_dense2=True,
                    shared_pages=shared_pages,
                    interpret=True,
                )
                ref = _exact_decode_ref(
                    np.asarray(q_bf16),
                    k_new_u8,
                    v_new_u8,
                    stock_cache_u8,
                    kv_lens,
                    page_indices.ravel(),
                    n_act,
                    page_size,
                    sm_scale,
                    pps,
                )
                out_np = np.asarray(out_j, dtype=np.float32)
                self.assertGreater(_cosine(out_np[:n_act], ref[:n_act]), min_cos)
                if n_act < bs:
                    self.assertEqual(np.max(np.abs(out_np[n_act:])), 0.0)

        # Short-context nod2 & d2 batchfull (0x7F NaN-poisoned Full-Attn via swabatch gears)
        for tag, use_d2, sw, pps, mode in (
            ("nod2_swa_w8", False, 2048, 8, "w8,c1024"),
            ("nod2_full_batchfull", False, None, 8, "w8,c1024,batchfull"),
            ("d2_full_batchfull", True, None, 8, "w8,c1024,batchfull"),
        ):
            with self.subTest(case=tag):
                bs, n_act, page_size = 16, 13, 128
                n_phys_pages = 256
                stock_cache_u8 = np.full(
                    (n_phys_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8
                )
                stock_cache_u8[:, :, 0, :2, :] = _rand_fp8_u8(
                    rng, (n_phys_pages, page_size, 2, 128)
                )
                cache_u8 = (
                    _pack_stock_cache_to_dense2_u8(stock_cache_u8, page_size)
                    if use_d2
                    else stock_cache_u8
                )
                page_indices = np.arange(1, 1 + bs * pps, dtype=np.int32).reshape(bs, pps)
                kv_lens = np.zeros((bs,), dtype=np.int32)
                for s in range(n_act):
                    kv_lens[s] = 180 + (s * 29) % 300
                q_bf16 = jnp.asarray(
                    rng.standard_normal((bs, 16, 128)) * 0.35, dtype=jnp.bfloat16
                )
                k_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                v_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                out_j, _ = run_decode_fast_rpa(
                    q_bf16,
                    _u8_to_fp8(k_new_u8)[:, None, :],
                    _u8_to_fp8(v_new_u8)[:, None, :],
                    _u8_to_fp8(cache_u8),
                    jnp.asarray(kv_lens, dtype=jnp.int32),
                    jnp.asarray(page_indices.ravel(), dtype=jnp.int32),
                    jnp.arange(bs + 1, dtype=jnp.int32),
                    jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size),
                    jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32),
                    sm_scale=sm_scale,
                    sliding_window=sw,
                    k_scale=1.0,
                    v_scale=1.0,
                    mode_str=mode,
                    decode_query_size=1,
                    fused_out_pack=1,
                    kv_pack_dense2=use_d2,
                    interpret=True,
                )
                ref = _exact_decode_ref(
                    np.asarray(q_bf16),
                    k_new_u8,
                    v_new_u8,
                    stock_cache_u8,
                    kv_lens,
                    page_indices.ravel(),
                    n_act,
                    page_size,
                    sm_scale,
                    pps,
                )
                out_np = np.asarray(out_j, dtype=np.float32)
                self.assertGreater(_cosine(out_np[:n_act], ref[:n_act]), 0.999)
                self.assertEqual(np.max(np.abs(out_np[n_act:])), 0.0)

        # Page size 256 SWA (w_tot=9*256=2304) zero-concat dense2 path
        with self.subTest(case="swa_ps256_d2"):
            bs, n_act, page_size, sw = 16, 14, 256, 2048
            pps = 12
            n_phys_pages = 256
            stock_cache_u8 = np.full(
                (n_phys_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8
            )
            stock_cache_u8[:, :, 0, :2, :] = _rand_fp8_u8(
                rng, (n_phys_pages, page_size, 2, 128)
            )
            cache_u8 = _pack_stock_cache_to_dense2_u8(stock_cache_u8, page_size)
            page_indices = np.arange(1, 1 + bs * pps, dtype=np.int32).reshape(bs, pps)
            kv_lens = np.zeros((bs,), dtype=np.int32)
            for s in range(n_act):
                kv_lens[s] = 2200 + s * 17
            q_bf16 = jnp.asarray(
                rng.standard_normal((bs, 16, 128)) * 0.35, dtype=jnp.bfloat16
            )
            k_new_u8 = _rand_fp8_u8(rng, (bs, 128))
            v_new_u8 = _rand_fp8_u8(rng, (bs, 128))
            out_j, _ = run_decode_fast_rpa(
                q_bf16,
                _u8_to_fp8(k_new_u8)[:, None, :],
                _u8_to_fp8(v_new_u8)[:, None, :],
                _u8_to_fp8(cache_u8),
                jnp.asarray(kv_lens, dtype=jnp.int32),
                jnp.asarray(page_indices.ravel(), dtype=jnp.int32),
                jnp.arange(bs + 1, dtype=jnp.int32),
                jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size),
                jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32),
                sm_scale=sm_scale,
                sliding_window=sw,
                k_scale=1.0,
                v_scale=1.0,
                mode_str="w8,c1024,pbf16",
                decode_query_size=1,
                fused_out_pack=1,
                kv_pack_dense2=True,
                interpret=True,
            )
            ref = np.zeros((bs, 16, 128), dtype=np.float32)
            for s in range(n_act):
                kvl = int(kv_lens[s])
                first_k = max(0, kvl - sw)
                k_list = [
                    _fp8_from_u8(
                        stock_cache_u8[
                            int(page_indices[s, p // page_size]), p % page_size, 0, 0, :
                        ]
                    )
                    for p in range(first_k, kvl - 1)
                ] + [_fp8_from_u8(k_new_u8[s])]
                v_list = [
                    _fp8_from_u8(
                        stock_cache_u8[
                            int(page_indices[s, p // page_size]), p % page_size, 0, 1, :
                        ]
                    )
                    for p in range(first_k, kvl - 1)
                ] + [_fp8_from_u8(v_new_u8[s])]
                lg = (np.asarray(q_bf16[s], dtype=np.float32) @ np.stack(k_list).T) * sm_scale
                lg -= np.max(lg, axis=1, keepdims=True)
                pr = np.exp(lg)
                pr /= np.sum(pr, axis=1, keepdims=True)
                ref[s] = pr @ np.stack(v_list)
            out_np = np.asarray(out_j, dtype=np.float32)
            self.assertGreater(_cosine(out_np[:n_act], ref[:n_act]), 0.999)
            self.assertEqual(np.max(np.abs(out_np[n_act:])), 0.0)

    def test_mosaic_tpu7x_export_long_context_272k(self):
        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")

        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            for tag, pps, ps, use_d2, mode, n_grp, exp_sub in (
                ("ring4k_272k_ps128_d2", 2177, 128, True, "w8,c1024,ring4k", 0, "ring4k-mp32"),
                ("ring8k_272k_ps128_d2", 2177, 128, True, "w8,c1024,ring8k", 0, "ring4k-mp64"),
                ("ring4k_272k_ps256_d2", 1089, 256, True, "w8,c1024,ring4k", 0, "ring4k-mp16"),
                ("shared16_272k_ps128_d2", 2177, 128, True, "w8,c1024,shared16,pbf16", 4, "shared16-mp16"),
                ("shared32_272k_ps128_d2", 2177, 128, True, "w8,c1024,shared32,pbf16", 2, "shared32-mp16"),
                ("shared64_272k_ps128_d2", 2177, 128, True, "w8,c1024,shared64,pbf16", 1, "shared64-mp16"),
                ("shared64_272k_ps256_d2", 1089, 256, True, "w8,c1024,shared64,pbf16", 1, "shared64-mp8"),
                ("shared64_272k_ps256_q512_d2", 1089, 256, True, "w8,c1024,shared64,q512", 1, "shared64-mp8"),
                ("shared64_272k_ps256_c4096_d2", 1089, 256, True, "w8,c4096,shared64", 1, "shared64-mp16"),
                ("shared64_272k_ps256_shsplit_d2", 1089, 256, True, "w8,c4096,shared64,q512,shsplit,pbf16", 1, "shsplit"),
                ("shared64_272k_ps256_shsplit_shfp8_d2", 1089, 256, True, "w8,c4096,shared64,q512,shsplit,shfp8", 1, "shfp8"),
                ("shared64_272k_ps256_c2048_shfp8_d2", 1089, 256, True, "w8,c2048,shared64,q256,shsplit,shfp8", 1, "shfp8"),
                ("shared64_272k_ps256_q128_shsplit_d2", 1089, 256, True, "w8,c4096,shared64,q128,shsplit,pbf16", 1, "q128"),
            ):
                with self.subTest(export=tag):
                    bs = 64
                    phys_pg = ps // 2 if use_d2 else ps
                    dist_len = 3 + n_grp if n_grp > 0 else 3
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q, kv, g, c, kl, pi, cq, ckv, d, _m=mode, _d2=use_d2: ragged_paged_attention(
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
                                sliding_window=None,
                                k_scale=1.0,
                                v_scale=1.0,
                                decode_query_size=1,
                                _fused_out_pack=2,
                                _prepared_inputs=True,
                                _skip_mixed=True,
                                _decode_fast=_m,
                                _decode_out_norm_eps=1e-5,
                                _decode_gate_in_values=True,
                                _kv_pack_dense2=_d2,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, bs, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs, 1024), jnp.bfloat16),
                        jax.ShapeDtypeStruct((4096, phys_pg, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs,), jnp.int32),
                        jax.ShapeDtypeStruct((bs * pps,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((dist_len,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("RPAd_fast", hlo)
                    self.assertIn(exp_sub, hlo)

            # Short-context nod2 batchfull Mosaic exports (1/1k & 8k/1k)
            for tag, pps, sw, use_d2, mode, exp_sub in (
                ("nod2_11k_swa_w8", 8, 2048, False, "w8,c1024", "swabatch"),
                ("nod2_11k_full_batchfull", 8, None, False, "w8,c1024,batchfull", "batchfull"),
                ("nod2_8k1k_swa_w8", 70, 2048, False, "w8,c1024", "swabatch"),
                ("nod2_8k1k_full_dyn2k", 136, None, False, "w8,c1024,batchfull,dyn2k", "c2048-streamdyn"),
                ("nod2_8k1k_full_dyn4k", 136, None, False, "w8,c1024,batchfull,dyn4k", "c4096-streamdyn"),
            ):
                with self.subTest(export=tag):
                    bs = 64
                    phys_pg = 64 if use_d2 else 128
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q, kv, g, c, kl, pi, cq, ckv, d, _sw=sw, _m=mode, _d2=use_d2: ragged_paged_attention(
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
                                decode_query_size=1,
                                _fused_out_pack=2,
                                _prepared_inputs=True,
                                _skip_mixed=True,
                                _decode_fast=_m,
                                _decode_out_norm_eps=1e-5,
                                _decode_gate_in_values=True,
                                _kv_pack_dense2=_d2,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, bs, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs, 1024), jnp.bfloat16),
                        jax.ShapeDtypeStruct((4096, phys_pg, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs,), jnp.int32),
                        jax.ShapeDtypeStruct((bs * pps,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("RPAd_fast", hlo)
                    self.assertIn(exp_sub, hlo)


    def test_8k1k_dyn2k_dyn4k_full_attn_parity_interpret(self):
        """Verify 8k/1k Full-Attn dyn2k/dyn4k routing and 0x7F-poisoned parity across boundary lengths."""
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            parse_decode_fast_config,
            run_decode_fast_rpa,
        )

        # Config routing assertions: c1024 must not clobber dyn2k/dyn4k on 8k/1k (pps=70..136).
        for pps in (70, 136):
            cfg2k = parse_decode_fast_config(
                "w8,c1024,batchfull,dyn2k",
                sliding_window=None,
                pages_per_seq=pps,
                page_size=128,
            )
            self.assertEqual(cfg2k.loop_kind, "streamdyn")
            self.assertEqual(cfg2k.chunk_sz, 2048)
            cfg4k = parse_decode_fast_config(
                "w8,c1024,batchfull,dyn4k",
                sliding_window=None,
                pages_per_seq=pps,
                page_size=128,
            )
            self.assertEqual(cfg4k.loop_kind, "streamdyn")
            self.assertEqual(cfg4k.chunk_sz, 4096)
            # SWA remains swabatch/c1024 even when dyn2k is in the mode string.
            cfg_swa = parse_decode_fast_config(
                "w8,c1024,batchfull,dyn2k",
                sliding_window=2048,
                pages_per_seq=pps,
                page_size=128,
            )
            self.assertEqual(cfg_swa.loop_kind, "swabatch")
            self.assertEqual(cfg_swa.chunk_sz, 1024)
        # 1/1k (pps <= 32) Full-Attn remains swabatch/c1024 when batchfull is set.
        cfg_11k = parse_decode_fast_config(
            "w8,c1024,batchfull,dyn2k",
            sliding_window=None,
            pages_per_seq=8,
            page_size=128,
        )
        self.assertEqual(cfg_11k.loop_kind, "swabatch")
        self.assertEqual(cfg_11k.chunk_sz, 1024)

        rng = np.random.default_rng(1947)
        sm_scale = 0.125
        bs, n_act, page_size, pps = 16, 14, 128, 70
        n_phys_pages = bs * pps + 4
        stock_cache_u8 = np.full(
            (n_phys_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8
        )
        stock_cache_u8[:, :, 0, :2, :] = _rand_fp8_u8(
            rng, (n_phys_pages, page_size, 2, 128)
        )
        page_indices = np.arange(1, 1 + bs * pps, dtype=np.int32).reshape(bs, pps)
        kv_lens = np.array(
            [1, 97, 128, 129, 513, 2047, 2048, 2049, 4096, 8191, 8192, 8193, 8450, 8831, 0, 0],
            dtype=np.int32,
        )
        q_bf16 = jnp.asarray(
            rng.standard_normal((bs, 16, 128)) * 0.35, dtype=jnp.bfloat16
        )
        k_new_u8 = _rand_fp8_u8(rng, (bs, 128))
        v_new_u8 = _rand_fp8_u8(rng, (bs, 128))
        ref = _exact_decode_ref(
            np.asarray(q_bf16),
            k_new_u8,
            v_new_u8,
            stock_cache_u8,
            kv_lens,
            page_indices.ravel(),
            n_act,
            page_size,
            sm_scale,
            pps,
        )

        with jax.default_device(jax.devices("cpu")[0]):
            for tag, use_d2, mode in (
                ("nod2_dyn2k", False, "w8,c1024,batchfull,dyn2k"),
                ("d2_dyn4k", True, "w8,c1024,batchfull,dyn4k"),
            ):
                with self.subTest(case=tag):
                    cache_u8 = (
                        _pack_stock_cache_to_dense2_u8(stock_cache_u8, page_size)
                        if use_d2
                        else stock_cache_u8
                    )
                    out_j, _ = run_decode_fast_rpa(
                        q_bf16,
                        _u8_to_fp8(k_new_u8)[:, None, :],
                        _u8_to_fp8(v_new_u8)[:, None, :],
                        _u8_to_fp8(cache_u8),
                        jnp.asarray(kv_lens, dtype=jnp.int32),
                        jnp.asarray(page_indices.ravel(), dtype=jnp.int32),
                        jnp.arange(bs + 1, dtype=jnp.int32),
                        jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size),
                        jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32),
                        sm_scale=sm_scale,
                        sliding_window=None,
                        k_scale=1.0,
                        v_scale=1.0,
                        mode_str=mode,
                        decode_query_size=1,
                        fused_out_pack=1,
                        kv_pack_dense2=use_d2,
                        interpret=True,
                    )
                    out_np = np.asarray(out_j, dtype=np.float32)
                    self.assertGreater(_cosine(out_np[:n_act], ref[:n_act]), 0.999)
                    self.assertEqual(np.max(np.abs(out_np[n_act:])), 0.0)


    def test_page_size_512_routing_interpret_and_mosaic(self):
        """Verify page_size=512 (d2 phys=256 / nod2 phys=512) routing, 0x7F interpret parity, and TPU7x export."""
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            can_use_decode_fast,
            parse_decode_fast_config,
            run_decode_fast_rpa,
        )

        q_3d = jnp.zeros((64, 16, 128), dtype=jnp.bfloat16)
        c_512_d2 = jnp.zeros((16, 256, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        c_512_nod2 = jnp.zeros((16, 512, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        for d2, c_buf in ((True, c_512_d2), (False, c_512_nod2)):
            for pps in (4, 18, 545):
                self.assertTrue(
                    can_use_decode_fast(
                        mode_str="w8,c1024,batchfull,dyn2k",
                        q=q_3d,
                        kv_cache_fused=c_buf,
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
                        kv_pack_dense2=d2,
                        sliding_window=None,
                        pages_per_seq=pps,
                        need_fused_post=True,
                    )
                )

        # At page_size=512: 1/1k (pps=4) uses swabatch; 8k/1k (pps=18) must NOT fall into 18-page swabatch!
        cfg_11k = parse_decode_fast_config(
            "w8,c1024,batchfull,dyn2k",
            sliding_window=None,
            pages_per_seq=4,
            page_size=512,
            kv_pack_dense2=True,
        )
        self.assertEqual(cfg_11k.loop_kind, "swabatch")
        cfg_8k1k = parse_decode_fast_config(
            "w8,c1024,batchfull,dyn2k",
            sliding_window=None,
            pages_per_seq=18,
            page_size=512,
            kv_pack_dense2=True,
        )
        self.assertEqual(cfg_8k1k.loop_kind, "streamdyn")
        self.assertEqual(cfg_8k1k.chunk_sz, 2048)
        cfg_256k = parse_decode_fast_config(
            "w8,c1024,batchfull,dyn2k",
            sliding_window=None,
            pages_per_seq=545,
            page_size=512,
            kv_pack_dense2=True,
        )
        self.assertEqual(cfg_256k.loop_kind, "ring4k")

        rng = np.random.default_rng(512)
        sm_scale = 0.125
        page_size = 512
        bs, n_act = 16, 12
        with jax.default_device(jax.devices("cpu")[0]):
            for tag, pps, sw, use_d2, mode, kv_fn in (
                (
                    "ps512_d2_11k_swa",
                    6,
                    2048,
                    True,
                    "w8,c1024,batchfull",
                    lambda i: 2100 + i * 37,
                ),
                (
                    "ps512_d2_11k_full",
                    4,
                    None,
                    True,
                    "w8,c1024,batchfull,dyn2k",
                    lambda i: 300 + i * 131,
                ),
                (
                    "ps512_d2_8k1k_dyn2k",
                    18,
                    None,
                    True,
                    "w8,c1024,batchfull,dyn2k",
                    lambda i: 257 + i * 791,
                ),
                (
                    "ps512_nod2_8k1k_dyn2k",
                    18,
                    None,
                    False,
                    "w8,c1024,batchfull,dyn2k",
                    lambda i: 513 + i * 743,
                ),
            ):
                with self.subTest(case=tag):
                    n_phys = bs * pps + 4
                    stock_u8 = np.full(
                        (n_phys, page_size, 1, 4, 128), 0x7F, dtype=np.uint8
                    )
                    stock_u8[:, :, 0, :2, :] = _rand_fp8_u8(
                        rng, (n_phys, page_size, 2, 128)
                    )
                    cache_u8 = (
                        _pack_stock_cache_to_dense2_u8(stock_u8, page_size)
                        if use_d2
                        else stock_u8
                    )
                    page_indices = np.arange(1, 1 + bs * pps, dtype=np.int32).reshape(
                        bs, pps
                    )
                    kv_lens = np.zeros((bs,), dtype=np.int32)
                    for i in range(n_act):
                        kv_lens[i] = min(kv_fn(i), pps * page_size)
                    q_bf16 = jnp.asarray(
                        rng.standard_normal((bs, 16, 128)) * 0.35, dtype=jnp.bfloat16
                    )
                    k_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                    v_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                    out_j, _ = run_decode_fast_rpa(
                        q_bf16,
                        _u8_to_fp8(k_new_u8)[:, None, :],
                        _u8_to_fp8(v_new_u8)[:, None, :],
                        _u8_to_fp8(cache_u8),
                        jnp.asarray(kv_lens, dtype=jnp.int32),
                        jnp.asarray(page_indices.ravel(), dtype=jnp.int32),
                        jnp.arange(bs + 1, dtype=jnp.int32),
                        jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size),
                        jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32),
                        sm_scale=sm_scale,
                        sliding_window=sw,
                        k_scale=1.0,
                        v_scale=1.0,
                        mode_str=mode,
                        decode_query_size=1,
                        fused_out_pack=1,
                        kv_pack_dense2=use_d2,
                        interpret=True,
                    )
                    if sw is None:
                        ref = _exact_decode_ref(
                            np.asarray(q_bf16),
                            k_new_u8,
                            v_new_u8,
                            stock_u8,
                            kv_lens,
                            page_indices.ravel(),
                            n_act,
                            page_size,
                            sm_scale,
                            pps,
                        )
                    else:
                        ref = np.zeros((bs, 16, 128), dtype=np.float32)
                        for s in range(n_act):
                            kvl = int(kv_lens[s])
                            fk = max(0, kvl - sw)
                            k_l = [
                                _fp8_from_u8(
                                    stock_u8[
                                        int(page_indices[s, p // page_size]),
                                        p % page_size,
                                        0,
                                        0,
                                        :,
                                    ]
                                )
                                for p in range(fk, kvl - 1)
                            ] + [_fp8_from_u8(k_new_u8[s])]
                            v_l = [
                                _fp8_from_u8(
                                    stock_u8[
                                        int(page_indices[s, p // page_size]),
                                        p % page_size,
                                        0,
                                        1,
                                        :,
                                    ]
                                )
                                for p in range(fk, kvl - 1)
                            ] + [_fp8_from_u8(v_new_u8[s])]
                            lg = (
                                np.asarray(q_bf16[s], dtype=np.float32)
                                @ np.stack(k_l).T
                            ) * sm_scale
                            lg -= np.max(lg, axis=1, keepdims=True)
                            pr = np.exp(lg)
                            pr /= np.sum(pr, axis=1, keepdims=True)
                            ref[s] = pr @ np.stack(v_l)
                    out_np = np.asarray(out_j, dtype=np.float32)
                    self.assertGreater(_cosine(out_np[:n_act], ref[:n_act]), 0.999)
                    self.assertEqual(np.max(np.abs(out_np[n_act:])), 0.0)

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            for tag, pps, sw, use_d2, mode, exp_sub in (
                ("ps512_d2_11k_swa", 6, 2048, True, "w8,c1024,batchfull", "swabatch"),
                ("ps512_d2_11k_full", 4, None, True, "w8,c1024,batchfull,dyn2k", "swabatch"),
                ("ps512_d2_8k1k_dyn2k", 18, None, True, "w8,c1024,batchfull,dyn2k", "c2048-streamdyn"),
                ("ps512_d2_272k_ring4k", 545, None, True, "w8,c1024,batchfull,dyn2k", "ring4k-mp8"),
            ):
                with self.subTest(export=tag):
                    bs = 64
                    phys_pg = 256 if use_d2 else 512
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q, kv, g, c, kl, pi, cq, ckv, d, _sw=sw, _m=mode, _d2=use_d2: ragged_paged_attention(
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
                                decode_query_size=1,
                                _fused_out_pack=2,
                                _prepared_inputs=True,
                                _skip_mixed=True,
                                _decode_fast=_m,
                                _decode_out_norm_eps=1e-5,
                                _decode_gate_in_values=True,
                                _kv_pack_dense2=_d2,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, bs, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs, 1024), jnp.bfloat16),
                        jax.ShapeDtypeStruct((1024, phys_pg, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs,), jnp.int32),
                        jax.ShapeDtypeStruct((bs * pps,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("RPAd_fast", hlo)
                    self.assertIn(exp_sub, hlo)


    def test_page_size_512_swa_eviction_and_d2_nod2_nan_poisoned(self):
        """page_size=512 SWA handles rel_first==0 skip and rel_first>0 eviction with 0x7F poison."""
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
        rng = np.random.default_rng(1122)
        bs, n_act, page_size, sw = 16, 14, 512, 2048
        pps = 6
        sm_scale = 128 ** -0.5
        n_phys_pages = bs * pps + 4
        for tag, use_d2 in (("ps512_nod2_swa", False), ("ps512_d2_swa", True)):
            with self.subTest(case=tag):
                stock_cache_u8 = np.full(
                    (n_phys_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8
                )
                stock_cache_u8[:, :, 0, :2, :] = _rand_fp8_u8(
                    rng, (n_phys_pages, page_size, 2, 128)
                )
                cache_u8 = (
                    _pack_stock_cache_to_dense2_u8(stock_cache_u8, page_size)
                    if use_d2
                    else stock_cache_u8
                )
                page_indices = np.arange(1, 1 + bs * pps, dtype=np.int32).reshape(bs, pps)
                # Cover <512 (last_p==0), 513..2048 (rel_first==0, last_p>0), and >2049 (rel_first>0).
                kv_lens = np.zeros((bs,), dtype=np.int32)
                sample_lens = [17, 511, 512, 513, 1024, 1537, 2048, 2049, 2100, 2500, 2560, 2600, 900, 2200]
                kv_lens[:n_act] = sample_lens
                q_bf16 = jnp.asarray(
                    rng.standard_normal((bs, 16, 128)) * 0.35, dtype=jnp.bfloat16
                )
                k_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                v_new_u8 = _rand_fp8_u8(rng, (bs, 128))
                out_j, _ = run_decode_fast_rpa(
                    q_bf16,
                    _u8_to_fp8(k_new_u8)[:, None, :],
                    _u8_to_fp8(v_new_u8)[:, None, :],
                    _u8_to_fp8(cache_u8),
                    jnp.asarray(kv_lens, dtype=jnp.int32),
                    jnp.asarray(page_indices.ravel(), dtype=np.int32),
                    jnp.arange(bs + 1, dtype=jnp.int32),
                    jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size),
                    jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32),
                    sm_scale=sm_scale,
                    sliding_window=sw,
                    k_scale=1.0,
                    v_scale=1.0,
                    mode_str="w8,c1024,batchfull,dyn4k",
                    decode_query_size=1,
                    fused_out_pack=1,
                    kv_pack_dense2=use_d2,
                    interpret=True,
                )
                ref = np.zeros((bs, 16, 128), dtype=np.float32)
                for s in range(n_act):
                    kvl = int(kv_lens[s])
                    first_k = max(0, kvl - sw)
                    pids = page_indices[s]
                    k_l = [
                        _fp8_from_u8(
                            stock_cache_u8[pids[p // page_size], p % page_size, 0, 0, :]
                        )
                        for p in range(first_k, kvl - 1)
                    ] + [_fp8_from_u8(k_new_u8[s])]
                    v_l = [
                        _fp8_from_u8(
                            stock_cache_u8[pids[p // page_size], p % page_size, 0, 1, :]
                        )
                        for p in range(first_k, kvl - 1)
                    ] + [_fp8_from_u8(v_new_u8[s])]
                    lg = (
                        np.asarray(q_bf16[s], dtype=np.float32) @ np.stack(k_l).T
                    ) * sm_scale
                    lg -= np.max(lg, axis=1, keepdims=True)
                    pr = np.exp(lg)
                    pr /= np.sum(pr, axis=1, keepdims=True)
                    ref[s] = pr @ np.stack(v_l)
                out_np = np.asarray(out_j, dtype=np.float32)
                self.assertGreater(_cosine(out_np[:n_act], ref[:n_act]), 0.999)
                self.assertEqual(np.max(np.abs(out_np[n_act:])), 0.0)

    def test_native_tp16_full_and_swa_nan_poisoned_and_export(self):
        """TP16 (8, 64) native mode unlocks swabatch, dyn4k, and ring4k across all 62 layers."""
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import can_use_decode_fast, run_decode_fast_rpa
        from jax import export as jax_export
        from unittest.mock import patch

        rng = np.random.default_rng(2026)
        q_tp16 = jnp.zeros((16, 8, 64), dtype=jnp.bfloat16)
        q_tp8 = jnp.zeros((16, 16, 128), dtype=jnp.bfloat16)
        c_512 = jnp.zeros((16, 512, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        self.assertTrue(
            can_use_decode_fast(
                mode_str="tp16",
                q=q_tp16,
                kv_cache_fused=c_512,
                out_dtype=jnp.bfloat16,
                decode_query_size=1,
                custom_mask=None,
                attention_sink=None,
                soft_cap=None,
                xai_temperature_len=None,
                q_scale=None,
                actual_num_q_heads=8,
                actual_num_kv_heads=1,
                actual_head_dim=64,
                sliding_window=2048,
                pages_per_seq=6,
                pure_unpacked_decode=True,
                kv_pack_dense2=False,
            )
        )
        self.assertFalse(
            can_use_decode_fast(
                mode_str="tp16",
                q=q_tp16,
                kv_cache_fused=c_512,
                out_dtype=jnp.bfloat16,
                decode_query_size=1,
                custom_mask=None,
                attention_sink=None,
                soft_cap=None,
                xai_temperature_len=None,
                q_scale=None,
                actual_num_q_heads=8,
                actual_num_kv_heads=1,
                actual_head_dim=64,
                sliding_window=None,
                pages_per_seq=6,
                pure_unpacked_decode=True,
                kv_pack_dense2=False,
            )
        )
        for sw, pps in ((2048, 6), (None, 19)):
            self.assertTrue(
                can_use_decode_fast(
                    mode_str="tp16,w8,c1024,batchfull,dyn4k",
                    q=q_tp16,
                    kv_cache_fused=c_512,
                    out_dtype=jnp.bfloat16,
                    decode_query_size=1,
                    custom_mask=None,
                    attention_sink=None,
                    soft_cap=None,
                    xai_temperature_len=None,
                    q_scale=None,
                    actual_num_q_heads=8,
                    actual_num_kv_heads=1,
                    actual_head_dim=64,
                    sliding_window=sw,
                    pages_per_seq=pps,
                    pure_unpacked_decode=True,
                    kv_pack_dense2=False,
                )
            )
        self.assertTrue(
            can_use_decode_fast(
                mode_str="tp16,w8,c1024,batchfull,dyn4k",
                q=q_tp8,
                kv_cache_fused=c_512,
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
                pages_per_seq=19,
                pure_unpacked_decode=False,
                kv_pack_dense2=False,
            )
        )

        # 0x7F NaN-poisoned parity across SWA, Full short swabatch, dyn4k, and ring4k.
        sm_scale = 64 ** -0.5
        page_size = 512
        for tag, sw, pps, mode, lens in (
            ("tp16_swa_short", 2048, 6, "tp16,w8,c1024,batchfull,dyn4k", [17, 511, 512, 513, 1024, 900, 3, 256, 700, 1023, 64, 512, 999, 1]),
            ("tp16_swa_evict", 2048, 6, "tp16,w8,c1024,batchfull,dyn4k", [17, 512, 1024, 2048, 2049, 2500, 2600, 513, 2100, 2400, 999, 2055, 2560, 7]),
            ("tp16_full_11k", None, 3, "tp16,w8,c1024,batchfull,dyn4k", [1, 17, 511, 512, 513, 1024, 1200, 1400, 999, 777, 500, 1500, 1025, 42]),
            ("tp16_full_dyn4k", None, 10, "tp16,w8,c1024,batchfull,dyn4k", [12, 512, 1025, 2048, 3000, 4096, 4500, 5000, 4900, 2500, 1500, 4100, 5119, 3]),
            ("tp16_full_ring4k", None, 17, "tp16,w8,c4096,ring4k,batchfull", [19, 1025, 4096, 4097, 5500, 8192, 8193, 8500, 8600, 6000, 3000, 7000, 8400, 5]),
        ):
            with self.subTest(case=tag):
                bs, n_act = 16, 14
                n_phys_pages = bs * pps + 4
                cache_u8 = np.full(
                    (n_phys_pages, page_size, 1, 4, 128), 0x7F, dtype=np.uint8
                )
                # Valid rows have head_dim=64 padded with 0.0 in lanes 64:128.
                cache_u8[:, :, 0, :2, :] = 0
                cache_u8[:, :, 0, :2, :64] = _rand_fp8_u8(
                    rng, (n_phys_pages, page_size, 2, 64)
                )
                page_indices = np.arange(1, 1 + bs * pps, dtype=np.int32).reshape(bs, pps)
                kv_lens = np.zeros((bs,), dtype=np.int32)
                kv_lens[:n_act] = lens
                # Re-poison tail rows >= kv_len - 1 in each active sequence's last page to verify sanitizer.
                for s in range(n_act):
                    old_l = int(kv_lens[s]) - 1
                    last_p = old_l // page_size
                    rem = old_l % page_size
                    cache_u8[page_indices[s, last_p], rem:, 0, :2, :] = 0x7F
                    if last_p + 1 < pps:
                        cache_u8[page_indices[s, last_p + 1 :], :, 0, :2, :] = 0x7F

                q_bf16 = jnp.asarray(
                    rng.standard_normal((bs, 8, 64)) * 0.35, dtype=jnp.bfloat16
                )
                k_new_u8 = _rand_fp8_u8(rng, (bs, 64))
                v_new_u8 = _rand_fp8_u8(rng, (bs, 64))
                out_j, _ = run_decode_fast_rpa(
                    q_bf16,
                    _u8_to_fp8(k_new_u8)[:, None, :],
                    _u8_to_fp8(v_new_u8)[:, None, :],
                    _u8_to_fp8(cache_u8),
                    jnp.asarray(kv_lens, dtype=jnp.int32),
                    jnp.asarray(page_indices.ravel(), dtype=np.int32),
                    jnp.arange(bs + 1, dtype=jnp.int32),
                    jnp.arange(bs + 1, dtype=jnp.int32) * (pps * page_size),
                    jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32),
                    sm_scale=sm_scale,
                    sliding_window=sw,
                    k_scale=1.0,
                    v_scale=1.0,
                    mode_str=mode,
                    decode_query_size=1,
                    fused_out_pack=1,
                    kv_pack_dense2=False,
                    interpret=True,
                )
                ref = np.zeros((bs, 8, 64), dtype=np.float32)
                for s in range(n_act):
                    kvl = int(kv_lens[s])
                    first_k = 0 if sw is None else max(0, kvl - sw)
                    pids = page_indices[s]
                    k_l = [
                        _fp8_from_u8(
                            cache_u8[pids[p // page_size], p % page_size, 0, 0, :64]
                        )
                        for p in range(first_k, kvl - 1)
                    ] + [_fp8_from_u8(k_new_u8[s])]
                    v_l = [
                        _fp8_from_u8(
                            cache_u8[pids[p // page_size], p % page_size, 0, 1, :64]
                        )
                        for p in range(first_k, kvl - 1)
                    ] + [_fp8_from_u8(v_new_u8[s])]
                    lg = (
                        np.asarray(q_bf16[s], dtype=np.float32) @ np.stack(k_l).T
                    ) * sm_scale
                    lg -= np.max(lg, axis=1, keepdims=True)
                    pr = np.exp(lg)
                    pr /= np.sum(pr, axis=1, keepdims=True)
                    ref[s] = pr @ np.stack(v_l)
                out_np = np.asarray(out_j, dtype=np.float32)
                self.assertEqual(out_np.shape, (bs, 8, 64))
                self.assertTrue(np.all(np.isfinite(out_np)))
                self.assertGreater(_cosine(out_np[:n_act], ref[:n_act]), 0.999)
                self.assertEqual(np.max(np.abs(out_np[n_act:])), 0.0)

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            for tag, pps, sw, exp_sub in (
                ("tp16_swa", 6, 2048, "swabatch"),
                ("tp16_full_11k", 3, None, "swabatch"),
                ("tp16_full_8k1k", 19, None, "c4096-streamdyn"),
                ("tp16_full_256k", 513, None, "ring4k-mp8"),
            ):
                with self.subTest(export=tag):
                    bs = 64
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q, k, v, c, kl, pi, cq, ckv, d, _sw=sw: ragged_paged_attention(
                                q,
                                k,
                                v,
                                c,
                                kl,
                                pi,
                                cq,
                                ckv,
                                d,
                                None,
                                None,
                                sm_scale=64 ** -0.5,
                                sliding_window=_sw,
                                k_scale=1.0,
                                v_scale=1.0,
                                decode_query_size=1,
                                _fused_out_pack=1,
                                _prepared_inputs=False,
                                _skip_mixed=True,
                                _decode_fast="tp16,w8,c1024,batchfull,dyn4k",
                                _kv_pack_dense2=False,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((bs, 8, 64), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs, 1, 64), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs, 1, 64), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((1024, 512, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs,), jnp.int32),
                        jax.ShapeDtypeStruct((bs * pps,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("RPAd_fast", hlo)
                    self.assertIn("-tp16", hlo)
                    self.assertIn(exp_sub, hlo)

    def test_cand4_hoistkv_full_only_u4flat_and_w7auto_dyn2ku_nan_poisoned(self):
        """hoistkv stays Full-only, u4flat runs SWA 1/1K cleanly, and dyn2ku covers 5K..8K without NaNs."""
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            parse_decode_fast_config,
            run_decode_fast_rpa,
        )

        # hoistkv must apply only to Full layers (sliding_window is None), never SWA.
        cfg_full = parse_decode_fast_config(
            "w8,hoistkv,dyn2ku",
            sliding_window=None,
            pages_per_seq=20,
            page_size=512,
        )
        cfg_swa = parse_decode_fast_config(
            "w8,hoistkv,u4flat",
            sliding_window=2048,
            pages_per_seq=4,
            page_size=512,
        )
        self.assertTrue(cfg_full.hoist_new_kv)
        self.assertTrue(cfg_full.dyn2k_unroll)
        self.assertEqual(cfg_full.loop_kind, "w7auto")
        self.assertFalse(cfg_swa.hoist_new_kv)
        self.assertTrue(cfg_swa.u4_flat)
        self.assertEqual(cfg_swa.loop_kind, "swabatch")

        rng = np.random.default_rng(20264)
        ps = 512
        # 0x7F NaN-poisoned Full w7auto + dyn2ku across <=5K, 5K..8K (4x2K), and >8K on nod2 & d2.
        lens_full = [17, 512, 1024, 1025, 4096, 5120, 5121, 6144, 7500, 8192, 8193, 9000]
        bs_f, pps_f = len(lens_full), 20
        stock_u8 = np.full((bs_f * pps_f + 8, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        stock_u8[:, :, 0, :2, :] = _rand_fp8_u8(rng, (bs_f * pps_f + 8, ps, 2, 128))
        d2_u8 = _pack_stock_cache_to_dense2_u8(stock_u8, ps)
        q_f = jnp.asarray((rng.standard_normal((bs_f, 16, 128)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
        k_f = _u8_to_fp8(_rand_fp8_u8(rng, (bs_f, 1, 128)))
        v_f = _u8_to_fp8(_rand_fp8_u8(rng, (bs_f, 1, 128)))
        pi_f = jnp.arange(bs_f * pps_f, dtype=jnp.int32)
        cuq_f = jnp.arange(bs_f + 1, dtype=jnp.int32)
        cukv_f = jnp.arange(bs_f + 1, dtype=jnp.int32) * (pps_f * ps)
        dist_f = jnp.asarray([bs_f, bs_f, bs_f], dtype=jnp.int32)
        kvl_f = jnp.asarray(lens_full, dtype=jnp.int32)
        for d2, cache_u8 in ((False, stock_u8), (True, d2_u8)):
            with self.subTest(d2=d2):
                kvc = _u8_to_fp8(cache_u8)
                o_ref, _ = run_decode_fast_rpa(
                    q_f, k_f, v_f, kvc, kvl_f, pi_f, cuq_f, cukv_f, dist_f,
                    sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                    mode_str="w8", kv_pack_dense2=d2, interpret=True,
                )
                o_opt, _ = run_decode_fast_rpa(
                    q_f, k_f, v_f, kvc, kvl_f, pi_f, cuq_f, cukv_f, dist_f,
                    sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                    mode_str="w8,hoistkv,dyn2ku", kv_pack_dense2=d2, interpret=True,
                )
                self.assertTrue(np.all(np.isfinite(np.asarray(o_opt, dtype=np.float32))))
                self.assertGreater(_cosine(np.asarray(o_opt), np.asarray(o_ref)), 0.9999)

        # Also verify ps=256 streamdyn,dyn2ku tail (4096 < rel_end <= 5120 -> 5120 // 256 == 20 pages).
        with self.subTest(ps256_dyn2ku_5k=True):
            ps_256, pps_256 = 256, 36
            lens_256 = [4500, 5120, 6000, 8192] * 2
            bs_256 = len(lens_256)
            u8_256 = np.full((bs_256 * pps_256 + 8, ps_256, 1, 4, 128), 0x7F, dtype=np.uint8)
            u8_256[:, :, 0, :2, :] = _rand_fp8_u8(rng, (bs_256 * pps_256 + 8, ps_256, 2, 128))
            kvc_256 = _u8_to_fp8(u8_256)
            q_256 = jnp.asarray((rng.standard_normal((bs_256, 16, 128)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
            k_256 = _u8_to_fp8(_rand_fp8_u8(rng, (bs_256, 1, 128)))
            v_256 = _u8_to_fp8(_rand_fp8_u8(rng, (bs_256, 1, 128)))
            pi_256 = jnp.arange(bs_256 * pps_256, dtype=jnp.int32)
            cuq_256 = jnp.arange(bs_256 + 1, dtype=jnp.int32)
            cukv_256 = jnp.arange(bs_256 + 1, dtype=jnp.int32) * (pps_256 * ps_256)
            dist_256 = jnp.asarray([bs_256, bs_256, bs_256], dtype=jnp.int32)
            kvl_256 = jnp.asarray(lens_256, dtype=jnp.int32)
            o_256_ref, _ = run_decode_fast_rpa(
                q_256, k_256, v_256, kvc_256, kvl_256, pi_256, cuq_256, cukv_256, dist_256,
                sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,streamdyn,c2048", interpret=True,
            )
            o_256_opt, _ = run_decode_fast_rpa(
                q_256, k_256, v_256, kvc_256, kvl_256, pi_256, cuq_256, cukv_256, dist_256,
                sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,streamdyn,dyn2ku", interpret=True,
            )
            self.assertTrue(np.all(np.isfinite(np.asarray(o_256_opt, dtype=np.float32))))
            self.assertGreater(_cosine(np.asarray(o_256_opt), np.asarray(o_256_ref)), 0.9999)

        # 0x7F NaN-poisoned u4flat SWA 1/1K parity (1p, 2p, mixed).
        for tag, lens_s in (
            ("swa_1p", [512] * 16),
            ("swa_900", [900] * 16),
            ("swa_mix", [10, 512, 513, 900, 1024, 1025, 1800, 2048] * 2),
        ):
            with self.subTest(u4flat=tag):
                bs_s, pps_s = len(lens_s), 6
                s_u8 = np.full((bs_s * pps_s + 8, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
                s_u8[:, :, 0, :2, :] = _rand_fp8_u8(rng, (bs_s * pps_s + 8, ps, 2, 128))
                kvc_s = _u8_to_fp8(s_u8)
                q_s = jnp.asarray((rng.standard_normal((bs_s, 16, 128)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
                k_s = _u8_to_fp8(_rand_fp8_u8(rng, (bs_s, 1, 128)))
                v_s = _u8_to_fp8(_rand_fp8_u8(rng, (bs_s, 1, 128)))
                pi_s = jnp.arange(bs_s * pps_s, dtype=jnp.int32)
                cuq_s = jnp.arange(bs_s + 1, dtype=jnp.int32)
                cukv_s = jnp.arange(bs_s + 1, dtype=jnp.int32) * (pps_s * ps)
                dist_s = jnp.asarray([bs_s, bs_s, bs_s], dtype=jnp.int32)
                kvl_s = jnp.asarray(lens_s, dtype=jnp.int32)
                o_b, _ = run_decode_fast_rpa(
                    q_s, k_s, v_s, kvc_s, kvl_s, pi_s, cuq_s, cukv_s, dist_s,
                    sm_scale=128 ** -0.5, sliding_window=2048, k_scale=1.0, v_scale=1.0,
                    mode_str="w8", interpret=True,
                )
                o_u, _ = run_decode_fast_rpa(
                    q_s, k_s, v_s, kvc_s, kvl_s, pi_s, cuq_s, cukv_s, dist_s,
                    sm_scale=128 ** -0.5, sliding_window=2048, k_scale=1.0, v_scale=1.0,
                    mode_str="w8,u4flat", interpret=True,
                )
                self.assertTrue(np.all(np.isfinite(np.asarray(o_u, dtype=np.float32))))
                self.assertGreater(_cosine(np.asarray(o_u), np.asarray(o_b)), 0.99995)

    def test_cand4_tp16_prepared_qkv_epi_in_kernel_pack2_nan_poisoned_and_export(self):
        """TP=16 QKV_EPI (8, 1) prepared inputs unlock in_kernel_pack2 [B, 512] across SWA/Full/Shared16."""
        from sglang_tpu.attention_pallas_rpa_v3 import TpuPallasRpaV3AttnBackend
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            can_use_decode_fast,
            run_decode_fast_rpa,
        )

        # Verify TP=16 (8, 1) relaxation in can_forward_decode_prepared.
        impl = TpuPallasRpaV3AttnBackend.__new__(TpuPallasRpaV3AttnBackend)
        impl._dcp = 1
        impl._decode_prepared_op = object()
        impl.kv_cache_quant_dtype = torch.float8_e4m3fn
        impl.forward_metadata = types.SimpleNamespace(
            decode_query_size=1,
            cp_write_segments=None,
            kv_pack_dense2=False,
        )
        fb = types.SimpleNamespace(
            forward_mode=types.SimpleNamespace(
                is_decode=lambda: True,
                is_target_verify=lambda: False,
            )
        )
        layer_tp16 = types.SimpleNamespace(
            is_cross_attention=False,
            tp_q_head_num=8,
            tp_k_head_num=1,
            qk_head_dim=64,
            v_head_dim=64,
            logit_cap=0,
        )
        layer_bad = types.SimpleNamespace(
            is_cross_attention=False,
            tp_q_head_num=4,
            tp_k_head_num=1,
            qk_head_dim=64,
            v_head_dim=64,
            logit_cap=0,
        )
        self.assertTrue(impl.can_forward_decode_prepared(layer_tp16, fb))
        self.assertFalse(impl.can_forward_decode_prepared(layer_bad, fb))

        # Verify forward_decode_prepared handles both 2D [B, 1024] and 3D [B, 16, 128] fallback outputs.
        impl.forward_metadata.active_num_tokens = 4
        impl.forward_metadata.cu_q_lens = torch.zeros((5,), dtype=torch.int32)
        impl.forward_metadata.cu_kv_lens = torch.zeros((5,), dtype=torch.int32)
        impl._kv_lens = lambda _fb: torch.ones((4,), dtype=torch.int32)
        impl._layer_page_indices = lambda _lid, is_swa_layer=False: torch.zeros((4,), dtype=torch.int32)
        impl._layer_distribution = lambda is_swa_layer=False: torch.tensor([4, 4, 4], dtype=torch.int32)
        impl.token_to_kv_pool = types.SimpleNamespace(get_fused_kv_buffer=lambda _lid: (torch.zeros((1,)), 0))
        layer_tp16.layer_id = 0
        layer_tp16.sliding_window_size = None
        layer_tp16.k_scale_float = 1.0
        layer_tp16.v_scale_float = 1.0
        layer_tp16.scaling = 64 ** -0.5
        q_dummy = torch.zeros((1, 4, 8, 2, 128), dtype=torch.bfloat16)
        kv_dummy = torch.zeros((4, 1, 4, 128), dtype=torch.bfloat16)
        with (
            patch("sglang_tpu.attention_pallas_rpa_v3._fused_kv_operands", return_value=(torch.zeros((1,)), 0)),
            patch("sglang_tpu.attention_pallas_rpa_v3._unpack_out_heads", side_effect=lambda x, _p, d: x[..., :d]),
        ):
            impl._decode_prepared_op = lambda *a, **kw: torch.ones((4, 1024), dtype=torch.bfloat16)
            self.assertEqual(tuple(impl.forward_decode_prepared(q_dummy, kv_dummy, layer_tp16, fb).shape), (4, 512))
            impl._decode_prepared_op = lambda *a, **kw: torch.ones((4, 16, 128), dtype=torch.bfloat16)
            self.assertEqual(tuple(impl.forward_decode_prepared(q_dummy, kv_dummy, layer_tp16, fb).shape), (4, 512))

        rng = np.random.default_rng(20265)
        ps = 512
        for tag, lens, pps, sw, mode, sh_grp in (
            ("tp16_swa", [512, 900, 1025, 1, 513, 1024, 200, 777] * 2, 4, 2048, "tp16", 0),
            ("tp16_full_8k1k", [17, 512, 1500, 4096, 5120, 6144, 8192, 8500] * 2, 20, None, "tp16,hoistkv,dyn2ku", 0),
            ("tp16_shared16", [1025] * 16, 4, None, "shared16_host", 16),
        ):
            with self.subTest(tp16_prepared=tag):
                bs = len(lens)
                s_u8 = np.full((bs * pps + 8, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
                s_u8[:, :, 0, :2, :] = 0
                s_u8[:, :, 0, 0, :64] = _rand_fp8_u8(rng, (bs * pps + 8, ps, 64))
                s_u8[:, :, 0, 1, :64] = _rand_fp8_u8(rng, (bs * pps + 8, ps, 64))
                pi_np = (
                    np.broadcast_to(np.arange(pps, dtype=np.int32), (bs, pps)).copy()
                    if sh_grp
                    else np.arange(bs * pps, dtype=np.int32).reshape(bs, pps)
                )
                for s in range(bs):
                    old_l = int(lens[s]) - 1
                    lp = old_l // ps
                    rem = old_l % ps
                    s_u8[pi_np[s, lp], rem:, 0, :2, :] = 0x7F
                    if lp + 1 < pps:
                        s_u8[pi_np[s, lp + 1 :], :, 0, :2, :] = 0x7F
                kvc = _u8_to_fp8(s_u8)
                q16 = jnp.asarray((rng.standard_normal((bs, 8, 64)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
                k16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 64)))
                v16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 64)))
                gate = jnp.asarray((rng.standard_normal((bs, 512)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
                q_prep = jnp.pad(q16, ((0, 0), (0, 8), (0, 64))).reshape(1, bs, 8, 2, 128)
                kv_prep = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs, 1, 4, 128)
                kvl = jnp.asarray(lens, dtype=jnp.int32)
                pi = jnp.asarray(pi_np.ravel(), dtype=jnp.int32)
                cuq = jnp.arange(bs + 1, dtype=jnp.int32)
                cukv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps)
                dist = jnp.asarray([bs, bs, bs], dtype=jnp.int32)
                shp = jnp.asarray([(lens[0] - 1) // ps], dtype=jnp.int32) if sh_grp else None
                self.assertTrue(
                    can_use_decode_fast(
                        mode_str=mode, q=q_prep, kv_cache_fused=kvc, out_dtype=jnp.bfloat16,
                        decode_query_size=1, custom_mask=None, attention_sink=None, soft_cap=None,
                        xai_temperature_len=None, q_scale=None, actual_num_q_heads=16,
                        actual_num_kv_heads=1, actual_head_dim=128, sliding_window=sw,
                        pages_per_seq=pps, pure_unpacked_decode=False, kv_pack_dense2=False,
                        need_fused_post=True,
                    )
                )
                o_ref, _ = run_decode_fast_rpa(
                    q16, k16, v16, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=64 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                    mode_str="shared16_host" if sh_grp else "tp16",
                    fused_out_pack=1, prepared_inputs=False, interpret=True, shared_pages=shp,
                )
                xf = np.asarray(o_ref, dtype=np.float32)
                inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + 1e-6)
                out_ref = jnp.asarray(xf * inv, dtype=jnp.bfloat16).reshape(-1, 512) * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(jnp.bfloat16)
                o_prep, _ = run_decode_fast_rpa(
                    q_prep, kv_prep, None, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=64 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                    mode_str=mode, fused_out_pack=2, prepared_inputs=True,
                    gate=gate, out_norm_eps=1e-6, interpret=True, shared_pages=shp,
                )
                self.assertEqual(o_prep.shape, (bs, 512))
                self.assertTrue(np.all(np.isfinite(np.asarray(o_prep, dtype=np.float32))))
                self.assertGreater(_cosine(np.asarray(o_prep), np.asarray(out_ref)), 0.99995)

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import ragged_paged_attention
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            for tag, pps, sw, mode in (
                ("tp16_epi_swa", 4, 2048, "tp16,u4flat"),
                ("tp16_epi_8k1k", 20, None, "tp16,w8,hoistkv,dyn2ku"),
            ):
                with self.subTest(export=tag):
                    bs = 64
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q, kv, g, c, kl, pi, cq, ckv, d, _sw=sw, _m=mode: ragged_paged_attention(
                                q, kv, g, c, kl, pi, cq, ckv, d,
                                None, None, sm_scale=64 ** -0.5, sliding_window=_sw,
                                k_scale=1.0, v_scale=1.0, decode_query_size=1,
                                _fused_out_pack=2, _prepared_inputs=True, _skip_mixed=True,
                                _decode_fast=_m, _decode_out_norm_eps=1e-6,
                                _decode_gate_in_values=True, _decode_tp16_prepared=True,
                                _kv_pack_dense2=False,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, bs, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs, 512), jnp.bfloat16),
                        jax.ShapeDtypeStruct((1024, 512, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs,), jnp.int32),
                        jax.ShapeDtypeStruct((bs * pps,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("RPAd_fast", hlo)
                    self.assertIn("-pk2-onorm-gate", hlo)
                    self.assertIn("-tp16", hlo)

            # Also verify ungated _decode_tp16_prepared=True exports cleanly (Shared16 [64, 512]).
            hlo_sh = jax_export.export(
                jax.jit(
                    lambda q, kv, c, kl, pi, cq, ckv, d: ragged_paged_attention(
                        q, kv, None, c, kl, pi, cq, ckv, d,
                        None, None, sm_scale=64 ** -0.5, sliding_window=None,
                        k_scale=1.0, v_scale=1.0, decode_query_size=1,
                        _fused_out_pack=2, _prepared_inputs=True, _skip_mixed=True,
                        _decode_fast="shared16_host", _decode_tp16_prepared=True,
                        _kv_pack_dense2=False,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, 64, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((64, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((1024, 512, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64,), jnp.int32),
                jax.ShapeDtypeStruct((64 * 4,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((7,), jnp.int32),
            ).mlir_module()
            self.assertIn("RPAd_fast", hlo_sh)
            self.assertIn("-tp16", hlo_sh)

    def test_cand4_tp16_backend_init_pack1_and_fake_impl_shapes(self):
        """Backend __init__ builds _decode_prepared* ops for pack==1 (1KV d=64), FakeTensor matches [B, 512]."""
        from torch._subclasses.fake_tensor import FakeTensorMode
        from sglang.srt.runtime_context import get_context
        from sglang_tpu.attention_pallas_rpa_v3 import TpuPallasRpaV3AttnBackend
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
        from sglang_tpu.envs import envs
        from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

        ctx_override = get_context().override_server_args(
            speculative_num_draft_tokens=None,
            speculative_algorithm=None,
            speculative_num_steps=None,
            speculative_eagle_topk=1,
            page_size=512,
        )
        ctx_override.install()
        self.addCleanup(ctx_override.restore)

        def _make_runner(kv_pack_heads: int, head_dim: int):
            pool = TpuFusedMHATokenToKVPool.__new__(TpuFusedMHATokenToKVPool)
            pool.kv_pack_heads = kv_pack_heads
            pool.head_num = 1
            pool.head_dim = head_dim
            pool.kv_pack_dense2 = False
            server_args = types.SimpleNamespace(
                speculative_num_draft_tokens=None,
                speculative_algorithm=None,
                speculative_num_steps=None,
                speculative_eagle_topk=1,
                page_size=512,
            )
            return types.SimpleNamespace(
                device=torch.device("cpu"),
                page_size=512,
                model_config=types.SimpleNamespace(
                    context_len=2048,
                    hf_text_config=types.SimpleNamespace(rms_norm_eps=1e-6),
                ),
                server_args=server_args,
                req_to_token_pool=types.SimpleNamespace(
                    req_to_token=torch.zeros(1, 2056, dtype=torch.int32)
                ),
                token_to_kv_pool=pool,
                is_hybrid_swa=False,
                kv_cache_dtype=torch.float8_e4m3fn,
                is_draft_worker=False,
                spec_algorithm=types.SimpleNamespace(is_none=lambda: True),
            )

        orig_custom_op = torch.library.custom_op

        def _cpu_custom_op(*args, **kwargs):
            kwargs["device_types"] = None
            return orig_custom_op(*args, **kwargs)

        with (
            patch("torch.library.custom_op", side_effect=_cpu_custom_op),
            envs.SGLANG_TPU_MUSE_DECODE_QKV_EPILOGUE.override(True),
            envs.SGLANG_TPU_MUSE_DECODE_ATTN_GATE_FUSE.override(True),
            envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.override(""),
            envs.SGLANG_TPU_RPA_V3_SKIP_EMPTY_DECODE.override(False),
            envs.SGLANG_TPU_RPA_V3_PREFILL_FUSED_OUT.override(False),
        ):
            be_tp16 = TpuPallasRpaV3AttnBackend(_make_runner(kv_pack_heads=1, head_dim=64))
            self.assertIsNotNone(be_tp16._decode_prepared_op)
            self.assertIsNotNone(be_tp16._decode_prepared_gated_op)

            be_tp8 = TpuPallasRpaV3AttnBackend(_make_runner(kv_pack_heads=2, head_dim=128))
            self.assertIsNotNone(be_tp8._decode_prepared_op)
            self.assertIsNotNone(be_tp8._decode_prepared_gated_op)

            with FakeTensorMode():
                bs = 64
                dev = "meta"
                q_prep = torch.empty((1, bs, 8, 2, 128), dtype=torch.bfloat16, device=dev)
                kv_prep = torch.empty((bs, 1, 4, 128), dtype=torch.float8_e4m3fn, device=dev)
                kvc = torch.empty((128, 512, 1, 4, 128), dtype=torch.float8_e4m3fn, device=dev)
                kvl = torch.empty((bs,), dtype=torch.int32, device=dev)
                pi = torch.empty((bs * 4,), dtype=torch.int32, device=dev)
                cuq = torch.empty((bs + 1,), dtype=torch.int32, device=dev)
                cukv = torch.empty((bs + 1,), dtype=torch.int32, device=dev)
                dist = torch.empty((3,), dtype=torch.int32, device=dev)
                gate_512 = torch.empty((bs, 512), dtype=torch.bfloat16, device=dev)
                gate_1024 = torch.empty((bs, 1024), dtype=torch.bfloat16, device=dev)

                out16_u = be_tp16._decode_prepared_op(
                    q_prep, kv_prep, kv_prep, kvc, kvl, pi, cuq, cukv, dist,
                    None, 0.125, None, None, 1.0, 1.0, 1, None, True, None,
                )
                out16_g = be_tp16._decode_prepared_gated_op(
                    q_prep, kv_prep, gate_512, kvc, kvl, pi, cuq, cukv, dist,
                    None, 0.125, None, None, 1.0, 1.0, 1, None, True, None,
                )
                out8_u = be_tp8._decode_prepared_op(
                    q_prep, kv_prep, kv_prep, kvc, kvl, pi, cuq, cukv, dist,
                    None, 0.125, None, None, 1.0, 1.0, 1, None, True, None,
                )
                out8_g = be_tp8._decode_prepared_gated_op(
                    q_prep, kv_prep, gate_1024, kvc, kvl, pi, cuq, cukv, dist,
                    None, 0.125, None, None, 1.0, 1.0, 1, None, True, None,
                )
                self.assertEqual(tuple(out16_u.shape), (bs, 512))
                self.assertEqual(tuple(out16_g.shape), (bs, 512))
                self.assertEqual(tuple(out8_u.shape), (bs, 16, 128))
                self.assertEqual(tuple(out8_g.shape), (bs, 1024))

        # Verify ungated force_native_tp16=True on shared16_host returns (bs, 512) cleanly.
        rng = np.random.default_rng(20266)
        bs_sh, pps_sh, ps_sh = 16, 4, 512
        s_u8 = np.full((bs_sh * pps_sh + 8, ps_sh, 1, 4, 128), 0x7F, dtype=np.uint8)
        s_u8[:, :, 0, :2, :] = 0
        s_u8[:, :, 0, 0, :64] = _rand_fp8_u8(rng, (bs_sh * pps_sh + 8, ps_sh, 64))
        s_u8[:, :, 0, 1, :64] = _rand_fp8_u8(rng, (bs_sh * pps_sh + 8, ps_sh, 64))
        kvc_sh = _u8_to_fp8(s_u8)
        q16 = jnp.asarray((rng.standard_normal((bs_sh, 8, 64)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
        k16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sh, 1, 64)))
        v16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sh, 1, 64)))
        q_p = jnp.pad(q16, ((0, 0), (0, 8), (0, 64))).reshape(1, bs_sh, 8, 2, 128)
        kv_p = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs_sh, 1, 4, 128)
        pi_sh = jnp.asarray(np.broadcast_to(np.arange(pps_sh, dtype=np.int32), (bs_sh, pps_sh)).ravel(), dtype=jnp.int32)
        o_sh, _ = run_decode_fast_rpa(
            q_p, kv_p, None, kvc_sh, jnp.full((bs_sh,), 1025, dtype=jnp.int32), pi_sh,
            jnp.arange(bs_sh + 1, dtype=jnp.int32), jnp.arange(bs_sh + 1, dtype=jnp.int32) * (pps_sh * ps_sh),
            jnp.asarray([bs_sh, bs_sh, bs_sh], dtype=jnp.int32),
            sm_scale=64 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str="shared16_host", fused_out_pack=2, prepared_inputs=True,
            force_native_tp16=True, interpret=True, shared_pages=jnp.asarray([2], dtype=jnp.int32),
        )
        self.assertEqual(o_sh.shape, (bs_sh, 512))
        self.assertTrue(np.all(np.isfinite(np.asarray(o_sh, dtype=np.float32))))

    def test_cand5_tp16_q8_all_gears_zero_ulp(self):
        """#2907/#2992: TP=16 q8=8 rows must be 0-ULP exact vs q16 (noq8) across all gears."""
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        rng = np.random.default_rng(20269)
        bs = 4
        n_act = 3  # exercises inactive zero-row path on every gear
        cases = [
            ("w7auto_le1k", 512, 4, None, [512, 900, 1024, 1], "tp16,w8,hoistkv,dyn2ku", False),
            ("u4flat_swa", 512, 2, 1024, [512, 900, 1024, 1], "tp16,w8,u4flat", False),
            ("stream1p_swa", 512, 5, 2048, [512, 2000, 2500, 1], "tp16,w8", False),
            ("dyn2ku_8k", 512, 16, None, [1024, 5120, 7500, 1], "tp16,w8,hoistkv,dyn2ku", False),
            ("streamdyn_5k", 512, 12, None, [1024, 3000, 5120, 1], "tp16,w8,hoistkv,streamdyn,c4096", False),
            ("ring4k_lc", 512, 24, None, [1024, 6000, 9500, 1], "tp16,w8,ring4k,c4096", False),
            ("subblock2k_gt16k", 128, 130, None, [200, 1000, 2049, 1], "tp16,stream0b", False),
            ("qepi_dyn2ku", 512, 16, None, [512, 5120, 7500, 1], "tp16,w8,hoistkv,dyn2ku", True),
        ]
        for tag, ps, pps, sw, lens, mode, prep in cases:
            with self.subTest(gear=tag):
                s_u8 = np.full((bs * pps + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
                pi_np = np.arange(bs * pps, dtype=np.int32).reshape(bs, pps)
                for s in range(n_act):
                    old = lens[s] - 1
                    lp = old // ps
                    rem = old % ps
                    if lp > 0:
                        s_u8[pi_np[s, :lp], :, 0, 0, :64] = _rand_fp8_u8(rng, (lp, ps, 64))
                        s_u8[pi_np[s, :lp], :, 0, 1, :64] = _rand_fp8_u8(rng, (lp, ps, 64))
                        s_u8[pi_np[s, :lp], :, 0, :2, 64:] = 0
                    if rem > 0:
                        s_u8[pi_np[s, lp], :rem, 0, 0, :64] = _rand_fp8_u8(rng, (rem, 64))
                        s_u8[pi_np[s, lp], :rem, 0, 1, :64] = _rand_fp8_u8(rng, (rem, 64))
                        s_u8[pi_np[s, lp], :rem, 0, :2, 64:] = 0
                kvc = _u8_to_fp8(s_u8)
                q16 = jnp.asarray((rng.standard_normal((bs, 8, 64)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
                k16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 64)))
                v16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 64)))
                gate = jnp.asarray((rng.standard_normal((bs, 512)) * 0.4).astype(np.float32), dtype=jnp.bfloat16) if prep else None
                q_in = jnp.pad(q16, ((0, 0), (0, 8), (0, 64))).reshape(1, bs, 8, 2, 128) if prep else q16
                k_in = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs, 1, 4, 128) if prep else k16
                v_in = None if prep else v16
                kvl = jnp.asarray(lens, dtype=jnp.int32)
                pi = jnp.asarray(pi_np.ravel(), dtype=np.int32)
                cuq = jnp.arange(bs + 1, dtype=jnp.int32)
                cukv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps)
                dist = jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32)

                o_q16, _ = run_decode_fast_rpa(
                    q_in, k_in, v_in, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=64 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                    mode_str=f"{mode},noq8", fused_out_pack=(2 if prep else 1),
                    prepared_inputs=prep, gate=gate, out_norm_eps=(1e-6 if prep else None), interpret=True,
                )
                o_q8, _ = run_decode_fast_rpa(
                    q_in, k_in, v_in, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=64 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                    mode_str=mode, fused_out_pack=(2 if prep else 1),
                    prepared_inputs=prep, gate=gate, out_norm_eps=(1e-6 if prep else None), interpret=True,
                )
                a16 = np.asarray(o_q16, dtype=np.float32)
                a8 = np.asarray(o_q8, dtype=np.float32)
                self.assertTrue(np.all(np.isfinite(a8)))
                self.assertEqual(float(np.max(np.abs(a8 - a16))), 0.0)

    def test_cand5_g4k_foldcol_gears_and_exports(self):
        """#2935/#2948: Opt-in g4k + foldcol (incl. 9216/10240, shared64, ring4k, d2 swa) 0x7F parity + export."""
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        rng = np.random.default_rng(20270)
        bs = 4
        # 1. TP=8 nod2 g4k (5K / 8K / 9216 / 10240 / >10240) + ring4k foldcol
        ps, pps = 512, 24
        lens = [5120, 8704, 9500, 11000]
        s_u8 = np.full((bs * pps + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        pi_np = np.arange(bs * pps, dtype=np.int32).reshape(bs, pps)
        for s, L in enumerate(lens):
            old = L - 1
            lp, rem = old // ps, old % ps
            if lp > 0:
                s_u8[pi_np[s, :lp], :, 0, 0, :] = _rand_fp8_u8(rng, (lp, ps, 128))
                s_u8[pi_np[s, :lp], :, 0, 1, :] = _rand_fp8_u8(rng, (lp, ps, 128))
            if rem > 0:
                s_u8[pi_np[s, lp], :rem, 0, 0, :] = _rand_fp8_u8(rng, (rem, 128))
                s_u8[pi_np[s, lp], :rem, 0, 1, :] = _rand_fp8_u8(rng, (rem, 128))
        kvc = _u8_to_fp8(s_u8)
        q = jnp.asarray((rng.standard_normal((bs, 16, 128)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
        kn = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 128)))
        vn = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 128)))
        gate = jnp.asarray((rng.standard_normal((bs, 1024)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
        kvl = jnp.asarray(lens, dtype=jnp.int32)
        pi = jnp.asarray(pi_np.ravel(), dtype=np.int32)
        cuq = jnp.arange(bs + 1, dtype=jnp.int32)
        cukv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps)
        dist = jnp.asarray([bs, bs, bs], dtype=jnp.int32)

        o_ref, _ = run_decode_fast_rpa(
            q, kn, vn, kvc, kvl, pi, cuq, cukv, dist,
            sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str="w8,hoistkv,dyn2ku", fused_out_pack=2, gate=gate, out_norm_eps=1e-6, interpret=True,
        )
        for m in ("w8,hoistkv,g4k", "w8,ring4k,c4096,foldcol"):
            o_g, _ = run_decode_fast_rpa(
                q, kn, vn, kvc, kvl, pi, cuq, cukv, dist,
                sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=m, fused_out_pack=2, gate=gate, out_norm_eps=1e-6, interpret=True,
            )
            self.assertTrue(np.all(np.isfinite(np.asarray(o_g, dtype=np.float32))))
            self.assertGreater(_cosine(np.asarray(o_g), np.asarray(o_ref)), 0.99999)

        # 2. shared64 foldcol
        bs_sh, pps_sh = 16, 4
        s_sh = np.full((bs_sh * pps_sh + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        s_sh[:, :, 0, 0, :] = _rand_fp8_u8(rng, (bs_sh * pps_sh + 4, ps, 128))
        s_sh[:, :, 0, 1, :] = _rand_fp8_u8(rng, (bs_sh * pps_sh + 4, ps, 128))
        q_sh = jnp.asarray((rng.standard_normal((bs_sh, 16, 128)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
        k_sh = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sh, 1, 128)))
        v_sh = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sh, 1, 128)))
        pi_sh = jnp.asarray(np.broadcast_to(np.arange(pps_sh, dtype=np.int32), (bs_sh, pps_sh)).ravel(), dtype=jnp.int32)
        o_sh_ref, _ = run_decode_fast_rpa(
            q_sh, k_sh, v_sh, _u8_to_fp8(s_sh), jnp.full((bs_sh,), 1025, dtype=jnp.int32), pi_sh,
            jnp.arange(bs_sh + 1, dtype=jnp.int32), jnp.arange(bs_sh + 1, dtype=jnp.int32) * (pps_sh * ps),
            jnp.asarray([bs_sh, bs_sh, bs_sh], dtype=jnp.int32),
            sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str="w8,hoistkv,shared64,c1024", interpret=True, shared_pages=jnp.asarray([2], dtype=jnp.int32),
        )
        o_sh_fc, _ = run_decode_fast_rpa(
            q_sh, k_sh, v_sh, _u8_to_fp8(s_sh), jnp.full((bs_sh,), 1025, dtype=jnp.int32), pi_sh,
            jnp.arange(bs_sh + 1, dtype=jnp.int32), jnp.arange(bs_sh + 1, dtype=jnp.int32) * (pps_sh * ps),
            jnp.asarray([bs_sh, bs_sh, bs_sh], dtype=jnp.int32),
            sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str="w8,hoistkv,shared64,c1024,foldcol", interpret=True, shared_pages=jnp.asarray([2], dtype=jnp.int32),
        )
        self.assertGreater(_cosine(np.asarray(o_sh_fc), np.asarray(o_sh_ref)), 0.999999)

        # 3. #3016 §1: Verify default record configs keep gear_4k=fold_128=fold_col=no_mask0=False & 0-ULP
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import parse_decode_fast_config

        for m, sw_c, pps_c, d2_c in (
            ("w8,hoistkv,dyn2ku", None, 24, False),
            ("w8,hoistkv,shared64,c1024", None, 4, False),
            ("w8,ring4k,c4096", None, 24, False),
            ("w8", 2048, 5, True),
        ):
            c_def = parse_decode_fast_config(
                m, sliding_window=sw_c, pages_per_seq=pps_c, page_size=ps, kv_pack_dense2=d2_c
            )
            self.assertFalse(c_def.gear_4k)
            self.assertFalse(c_def.fold_128)
            self.assertFalse(c_def.fold_col)
            self.assertFalse(c_def.no_mask0)

        s_d2 = np.full((bs * 5 + 4, ps // 2, 1, 4, 128), 0x7F, dtype=np.uint8)
        s_d2[: bs * 5, : 2500 // 2 - (2500 // ps) * (ps // 2) or (ps // 2), 0, :, :] = 0
        s_d2[:, :, 0, :, :] = _rand_fp8_u8(rng, (bs * 5 + 4, ps // 2, 4, 128))
        kvl_d2 = jnp.full((bs,), 2500, dtype=jnp.int32)
        pi_d2 = jnp.arange(bs * 5, dtype=np.int32)
        cukv_d2 = jnp.arange(bs + 1, dtype=jnp.int32) * (5 * ps)

        for tag_0, q_0, kn_0, vn_0, kvc_0, kvl_0, pi_0, cq_0, ckv_0, d_0, sw_0, m_0, d2_0, shp_0 in (
            ("dyn2ku", q, kn, vn, kvc, kvl, pi, cuq, cukv, dist, None, "w8,hoistkv,dyn2ku", False, None),
            ("ring4k", q, kn, vn, kvc, kvl, pi, cuq, cukv, dist, None, "w8,ring4k,c4096", False, None),
            (
                "shared64",
                q_sh,
                k_sh,
                v_sh,
                _u8_to_fp8(s_sh),
                jnp.full((bs_sh,), 1025, dtype=jnp.int32),
                pi_sh,
                jnp.arange(bs_sh + 1, dtype=jnp.int32),
                jnp.arange(bs_sh + 1, dtype=jnp.int32) * (pps_sh * ps),
                jnp.asarray([bs_sh, bs_sh, bs_sh], dtype=jnp.int32),
                None,
                "w8,hoistkv,shared64,c1024",
                False,
                jnp.asarray([2], dtype=jnp.int32),
            ),
            ("d2_swa", q, kn, vn, _u8_to_fp8(s_d2), kvl_d2, pi_d2, cuq, cukv_d2, dist, 2048, "w8", True, None),
        ):
            with self.subTest(default_0ulp=tag_0):
                o_a, _ = run_decode_fast_rpa(
                    q_0, kn_0, vn_0, kvc_0, kvl_0, pi_0, cq_0, ckv_0, d_0,
                    sm_scale=128 ** -0.5, sliding_window=sw_0, k_scale=1.0, v_scale=1.0,
                    mode_str=m_0, kv_pack_dense2=d2_0, interpret=True, shared_pages=shp_0,
                )
                o_b, _ = run_decode_fast_rpa(
                    q_0, kn_0, vn_0, kvc_0, kvl_0, pi_0, cq_0, ckv_0, d_0,
                    sm_scale=128 ** -0.5, sliding_window=sw_0, k_scale=1.0, v_scale=1.0,
                    mode_str=f"{m_0},nofold", kv_pack_dense2=d2_0, interpret=True, shared_pages=shp_0,
                )
                self.assertEqual(float(np.max(np.abs(np.asarray(o_a, dtype=np.float32) - np.asarray(o_b, dtype=np.float32)))), 0.0)

        # 4. TPU v7 export check for g4k-foldcol
        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import ragged_paged_attention
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            bs_e, pps_e = 64, 32
            hlo = jax_export.export(
                jax.jit(
                    lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: ragged_paged_attention(
                        q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_,
                        None, None, sm_scale=128 ** -0.5, sliding_window=None,
                        k_scale=1.0, v_scale=1.0, decode_query_size=1,
                        _fused_out_pack=2, _prepared_inputs=True, _skip_mixed=True,
                        _decode_fast="w8,hoistkv,g4k", _decode_out_norm_eps=1e-6,
                        _decode_gate_in_values=True, _kv_pack_dense2=False,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, bs_e, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((bs_e, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e, 1024), jnp.bfloat16),
                jax.ShapeDtypeStruct((bs_e * pps_e + 16, 512, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e * pps_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("-g4k-foldcol", hlo)

    def test_cand6_nod2_swa_fold128_and_pbf16_0ulp_and_export(self):
        """cand_w9_6: SWA w_pg=5 nod2 fold128 0-ULP parity, 0x7F Mixed NaN check, and TPU v7 export."""
        from jax import export as jax_export
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        rng = np.random.default_rng(606)
        bs, ps, pps = 8, 512, 20
        kv_lens_np = np.asarray([8192, 2500, 2049, 1500, 600, 1025, 2048, 8192], dtype=np.int32)
        s_u8 = np.full((bs * pps + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        rand_f8 = jax.device_get(
            jnp.asarray(rng.normal(0.0, 0.4, (bs * pps, ps, 2, 128)), jnp.float32).astype(jnp.float8_e4m3fn)
        )
        rand_u8 = np.frombuffer(rand_f8.tobytes(), dtype=np.uint8).reshape(bs, pps, ps, 2, 128)
        tok = np.arange(pps * ps).reshape(pps, ps)
        for b_i, kl in enumerate(kv_lens_np):
            vld = (tok < int(kl) - 1)[:, :, None]
            s_u8[b_i * pps : (b_i + 1) * pps, :, 0, 0, :] = np.where(vld, rand_u8[b_i, :, :, 0, :], np.uint8(0x7F))
            s_u8[b_i * pps : (b_i + 1) * pps, :, 0, 1, :] = np.where(vld, rand_u8[b_i, :, :, 1, :], np.uint8(0x7F))

        def _u8_to_fp8(arr_u8: np.ndarray) -> jax.Array:
            return jax.lax.bitcast_convert_type(jnp.asarray(arr_u8), jnp.float8_e4m3fn)

        q = jnp.asarray(rng.normal(0.0, 0.4, (bs, 16, 128)), jnp.bfloat16)
        kn = _u8_to_fp8(np.asarray(rng.integers(0, 64, (bs, 1, 128)), np.uint8))
        vn = _u8_to_fp8(np.asarray(rng.integers(0, 64, (bs, 1, 128)), np.uint8))
        pi = jnp.arange(bs * pps, dtype=jnp.int32)
        cuq = jnp.arange(bs + 1, dtype=jnp.int32)
        cukv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps)
        dist = jnp.asarray([bs, bs, bs], dtype=jnp.int32)
        ip = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")

        o_ref, _ = run_decode_fast_rpa(
            q, kn, vn, _u8_to_fp8(s_u8), jnp.asarray(kv_lens_np), pi, cuq, cukv, dist,
            sm_scale=128 ** -0.5, sliding_window=2048, k_scale=1.0, v_scale=1.0,
            mode_str="w8,g4k,nofold", kv_pack_dense2=False, interpret=ip,
        )
        o_f128, _ = run_decode_fast_rpa(
            q, kn, vn, _u8_to_fp8(s_u8), jnp.asarray(kv_lens_np), pi, cuq, cukv, dist,
            sm_scale=128 ** -0.5, sliding_window=2048, k_scale=1.0, v_scale=1.0,
            mode_str="w8,g4k", kv_pack_dense2=False, interpret=ip,
        )
        o_pbf, _ = run_decode_fast_rpa(
            q, kn, vn, _u8_to_fp8(s_u8), jnp.asarray(kv_lens_np), pi, cuq, cukv, dist,
            sm_scale=128 ** -0.5, sliding_window=2048, k_scale=1.0, v_scale=1.0,
            mode_str="w8,g4k,pbf16swa", kv_pack_dense2=False, interpret=ip,
        )
        o_full_ref, _ = run_decode_fast_rpa(
            q, kn, vn, _u8_to_fp8(s_u8), jnp.asarray(kv_lens_np), pi, cuq, cukv, dist,
            sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str="w8,g4k", kv_pack_dense2=False, interpret=ip,
        )
        o_full_swa, _ = run_decode_fast_rpa(
            q, kn, vn, _u8_to_fp8(s_u8), jnp.asarray(kv_lens_np), pi, cuq, cukv, dist,
            sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str="w8,g4k,pbf16swa", kv_pack_dense2=False, interpret=ip,
        )
        a_ref = np.asarray(o_ref, dtype=np.float32)
        a_f128 = np.asarray(o_f128, dtype=np.float32)
        a_pbf = np.asarray(o_pbf, dtype=np.float32)
        self.assertTrue(np.all(np.isfinite(a_ref)))
        self.assertEqual(float(np.max(np.abs(a_f128 - a_ref))), 0.0)
        self.assertEqual(
            float(np.max(np.abs(np.asarray(o_full_swa, dtype=np.float32) - np.asarray(o_full_ref, dtype=np.float32)))),
            0.0,
        )
        self.assertTrue(np.all(np.isfinite(a_pbf)))
        cos_pbf = float(np.dot(a_pbf.ravel(), a_ref.ravel()) / (np.linalg.norm(a_pbf) * np.linalg.norm(a_ref)))
        self.assertGreater(cos_pbf, 0.99999)

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            bs_e, pps_e = 64, 20
            hlo = jax_export.export(
                jax.jit(
                    lambda q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=128 ** -0.5, sliding_window=2048, k_scale=1.0, v_scale=1.0,
                        mode_str="w8,g4k,pbf16swa", kv_pack_dense2=False,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((bs_e, 16, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((bs_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e * pps_e + 16, 512, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e * pps_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("RPAd_fast-", hlo)
            self.assertIn("-pbf16-", hlo)
            hlo_full = jax_export.export(
                jax.jit(
                    lambda q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                        mode_str="w8,g4k,pbf16swa", kv_pack_dense2=False,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((bs_e, 16, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((bs_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e * pps_e + 16, 512, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e * pps_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertNotIn("-pbf16-", hlo_full)

    def test_cand7_ring4k_tail1k_2k_zero_ulp_and_export(self):
        """cand_w9_7: ring4k <=1K/<=2K/4K tails 0x7F NaN-poisoned parity (nod2+d2, Mixed) & TPU v7 export."""
        from jax import export as jax_export
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        rng = np.random.default_rng(707)
        bs, ps, pps = 8, 512, 512
        # Covers tail <= 1024 (500, 8704, 9215), 1024 < tail <= 2048 (1500, 6000, 9728), exact 4K (8192), >2K (7500).
        kv_lens_np = np.asarray([8704, 9728, 1500, 6000, 8192, 500, 7500, 9215], dtype=np.int32)
        n_pages = bs * 24 + 8
        q = jnp.asarray(rng.normal(0.0, 0.4, (bs, 16, 128)), jnp.bfloat16)
        kn = jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 64, (bs, 1, 128), np.uint8)), jnp.float8_e4m3fn)
        vn = jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 64, (bs, 1, 128), np.uint8)), jnp.float8_e4m3fn)
        pi_np = np.zeros(bs * pps, dtype=np.int32)
        for b_i in range(bs):
            pi_np[b_i * pps : b_i * pps + 20] = np.arange(b_i * 20, (b_i + 1) * 20, dtype=np.int32)
        pi = jnp.asarray(pi_np)
        cuq = jnp.arange(bs + 1, dtype=jnp.int32)
        cukv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps)
        dist = jnp.asarray([bs, bs, bs], dtype=jnp.int32)
        ip = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")

        rf8 = jax.device_get(
            jnp.asarray(rng.normal(0.0, 0.4, (bs * 20, ps, 2, 128)), jnp.float32).astype(jnp.float8_e4m3fn)
        )
        ru8 = np.frombuffer(rf8.tobytes(), dtype=np.uint8).reshape(bs, 20, ps, 2, 128)
        tok = np.arange(20 * ps).reshape(20, ps)

        for d2_flag in (False, True):
            phys_r = ps // 2 if d2_flag else ps
            kvc_u8 = np.full((n_pages, phys_r, 1, 4, 128), 0x7F, dtype=np.uint8)
            for b_i, kl in enumerate(kv_lens_np):
                vld = tok < (int(kl) - 1)
                p0 = b_i * 20
                if not d2_flag:
                    kvc_u8[p0 : p0 + 20, :, 0, 0, :] = np.where(vld[:, :, None], ru8[b_i, :, :, 0, :], np.uint8(0x7F))
                    kvc_u8[p0 : p0 + 20, :, 0, 1, :] = np.where(vld[:, :, None], ru8[b_i, :, :, 1, :], np.uint8(0x7F))
                else:
                    ke = np.where(vld[:, 0::2, None], ru8[b_i, :, 0::2, 0, :], np.uint8(0x7F))
                    ve = np.where(vld[:, 0::2, None], ru8[b_i, :, 0::2, 1, :], np.uint8(0x7F))
                    ko = np.where(vld[:, 1::2, None], ru8[b_i, :, 1::2, 0, :], np.uint8(0x7F))
                    vo = np.where(vld[:, 1::2, None], ru8[b_i, :, 1::2, 1, :], np.uint8(0x7F))
                    kvc_u8[p0 : p0 + 20, :, 0, 0, :] = ke
                    kvc_u8[p0 : p0 + 20, :, 0, 1, :] = ve
                    kvc_u8[p0 : p0 + 20, :, 0, 2, :] = ko
                    kvc_u8[p0 : p0 + 20, :, 0, 3, :] = vo
            kvc = jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8), jnp.float8_e4m3fn)
            o_r4k, _ = run_decode_fast_rpa(
                q, kn, vn, kvc, jnp.asarray(kv_lens_np), pi, cuq, cukv, dist,
                sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,g4k,foldcol", kv_pack_dense2=d2_flag, interpret=ip,
            )
            o_nofold, _ = run_decode_fast_rpa(
                q, kn, vn, kvc, jnp.asarray(kv_lens_np), pi, cuq, cukv, dist,
                sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,ring4k,nofold", kv_pack_dense2=d2_flag, interpret=ip,
            )
            # Exact ground-truth reference in fp32 per sequence using unpoisoned cached tokens + (kn, vn).
            kn_f = np.asarray(jax.lax.bitcast_convert_type(kn, jnp.float8_e4m3fn).astype(jnp.float32))
            vn_f = np.asarray(jax.lax.bitcast_convert_type(vn, jnp.float8_e4m3fn).astype(jnp.float32))
            q_f = np.asarray(q, dtype=np.float32)
            ref_rows = []
            for b_i, kl in enumerate(kv_lens_np):
                n_c = int(kl) - 1
                k_seq = np.concatenate(
                    [np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(ru8[b_i, :, :, 0, :]), jnp.float8_e4m3fn).astype(jnp.float32)).reshape(-1, 128)[:n_c], kn_f[b_i]],
                    axis=0,
                )
                v_seq = np.concatenate(
                    [np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(ru8[b_i, :, :, 1, :]), jnp.float8_e4m3fn).astype(jnp.float32)).reshape(-1, 128)[:n_c], vn_f[b_i]],
                    axis=0,
                )
                logits = (q_f[b_i] @ k_seq.T) * (128 ** -0.5)
                logits = logits - np.max(logits, axis=-1, keepdims=True)
                probs = np.exp(logits)
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
                ref_rows.append(np.asarray(jnp.asarray(probs @ v_seq, jnp.bfloat16), dtype=np.float32))
            a_ref = np.stack(ref_rows, axis=0)
            a_r4k = np.asarray(o_r4k, dtype=np.float32)
            a_nf = np.asarray(o_nofold, dtype=np.float32)
            with self.subTest(d2=d2_flag):
                self.assertTrue(np.all(np.isfinite(a_r4k)))
                self.assertEqual(float(np.max(np.abs(a_r4k - a_nf))), 0.0)
                cos_v = float(np.dot(a_r4k.ravel(), a_ref.ravel()) / (np.linalg.norm(a_r4k) * np.linalg.norm(a_ref)))
                self.assertGreater(cos_v, 0.99999)

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            bs_e, pps_e = 64, 512
            hlo = jax_export.export(
                jax.jit(
                    lambda q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=128 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                        mode_str="w8,g4k,foldcol", kv_pack_dense2=False,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((bs_e, 16, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((bs_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((4096, 512, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e * pps_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("ring4k", hlo)

    def test_cand10_swabatch_gears_dense2_verify_opts_and_export(self):
        """cand_w9_10: swabatch 1536/2048 gears + sequential d2 + DFlash verify opts (0x7F NaN-poisoned & TPU v7 export)."""
        import os

        from jax import export as jax_export
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
        from sglang_tpu.envs import envs

        rng = np.random.default_rng(1010)
        bs, ps, pps = 8, 512, 8
        pi = jnp.arange(bs * pps, dtype=jnp.int32)
        cukv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps)
        dist = jnp.asarray([bs, bs, bs], dtype=jnp.int32)
        ip = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")

        rf8 = jax.device_get(
            jnp.asarray(rng.normal(0.0, 0.4, (bs * pps, ps, 2, 128)), jnp.float32).astype(jnp.float8_e4m3fn)
        )
        ru8 = np.frombuffer(rf8.tobytes(), dtype=np.uint8).reshape(bs, pps, ps, 2, 128)
        tok = np.arange(pps * ps).reshape(pps, ps)
        n_pages = bs * pps + 4

        def _make_kvc(kv_lens_np, dq: int, d2: bool):
            phys_r = ps // 2 if d2 else ps
            kvc_u8 = np.full((n_pages, phys_r, 1, 4, 128), 0x7F, dtype=np.uint8)
            for b_i, kl in enumerate(kv_lens_np):
                vld = tok < (int(kl) - dq)
                p0 = b_i * pps
                if not d2:
                    kvc_u8[p0 : p0 + pps, :, 0, 0, :] = np.where(vld[:, :, None], ru8[b_i, :, :, 0, :], np.uint8(0x7F))
                    kvc_u8[p0 : p0 + pps, :, 0, 1, :] = np.where(vld[:, :, None], ru8[b_i, :, :, 1, :], np.uint8(0x7F))
                else:
                    kvc_u8[p0 : p0 + pps, :, 0, 0, :] = np.where(vld[:, 0::2, None], ru8[b_i, :, 0::2, 0, :], np.uint8(0x7F))
                    kvc_u8[p0 : p0 + pps, :, 0, 1, :] = np.where(vld[:, 0::2, None], ru8[b_i, :, 0::2, 1, :], np.uint8(0x7F))
                    kvc_u8[p0 : p0 + pps, :, 0, 2, :] = np.where(vld[:, 1::2, None], ru8[b_i, :, 1::2, 0, :], np.uint8(0x7F))
                    kvc_u8[p0 : p0 + pps, :, 0, 3, :] = np.where(vld[:, 1::2, None], ru8[b_i, :, 1::2, 1, :], np.uint8(0x7F))
            return jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8), jnp.float8_e4m3fn)

        def _fp32_ref(q_arr, kn_arr, vn_arr, kv_lens_np, dq: int, sw):
            q_f = np.asarray(q_arr, dtype=np.float32)
            kn_f = np.asarray(jax.lax.bitcast_convert_type(kn_arr, jnp.float8_e4m3fn).astype(np.float32))
            vn_f = np.asarray(jax.lax.bitcast_convert_type(vn_arr, jnp.float8_e4m3fn).astype(np.float32))
            refs = []
            for b_i, kl in enumerate(kv_lens_np):
                n_c = int(kl) - dq
                k_c = np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(ru8[b_i, :, :, 0, :]), jnp.float8_e4m3fn).astype(np.float32)).reshape(-1, 128)[:n_c]
                v_c = np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(ru8[b_i, :, :, 1, :]), jnp.float8_e4m3fn).astype(np.float32)).reshape(-1, 128)[:n_c]
                k_seq = np.concatenate([k_c, kn_f[b_i * dq : (b_i + 1) * dq, 0, :]], axis=0)
                v_seq = np.concatenate([v_c, vn_f[b_i * dq : (b_i + 1) * dq, 0, :]], axis=0)
                for qi in range(dq):
                    tot_k = n_c + qi + 1
                    first_k = max(0, tot_k - sw) if sw is not None else 0
                    logits = (q_f[b_i * dq + qi] @ k_seq[first_k:tot_k].T) * (128 ** -0.5)
                    logits -= np.max(logits, axis=-1, keepdims=True)
                    probs = np.exp(logits)
                    probs /= np.sum(probs, axis=-1, keepdims=True)
                    refs.append(np.asarray(jnp.asarray(probs @ v_seq[first_k:tot_k], jnp.bfloat16), dtype=np.float32))
            return np.stack(refs, axis=0)

        # 1. dq=1 swabatch 1536 (3p) and 2048 (4p) gears (nod2 + sequential d2, SWA & Full)
        for case_name, kl_vals, sw, d2_flag, mode_s in (
            ("swabatch_3p_d2_fc", [1050, 1100, 1200, 1300, 1400, 1500, 1536, 1025], 2049, True, "w8,noz,batchfull,g4k,foldcol"),
            ("swabatch_4p_nod2_u4", [1550, 1600, 1700, 1800, 1900, 2000, 2048, 1537], None, False, "w8,noz,batchfull,u4flat,g4k,foldcol"),
            ("swabatch_4p_d2_nf", [1550, 1600, 1700, 1800, 1900, 2000, 2048, 1537], 2049, True, "w8,noz,batchfull,nofold"),
        ):
            kl_np = np.asarray(kl_vals, dtype=np.int32)
            q = jnp.asarray(rng.normal(0.0, 0.4, (bs, 16, 128)), jnp.bfloat16)
            kn = jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 60, (bs, 1, 128), np.uint8)), jnp.float8_e4m3fn)
            vn = jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 60, (bs, 1, 128), np.uint8)), jnp.float8_e4m3fn)
            cuq = jnp.arange(bs + 1, dtype=jnp.int32)
            kvc = _make_kvc(kl_np, 1, d2_flag)
            out, _ = run_decode_fast_rpa(
                q, kn, vn, kvc, jnp.asarray(kl_np), pi, cuq, cukv, dist,
                sm_scale=128 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                mode_str=mode_s, kv_pack_dense2=d2_flag, interpret=ip,
            )
            a_out = np.asarray(out, dtype=np.float32)
            a_ref = _fp32_ref(q, kn, vn, kl_np, 1, sw)
            with self.subTest(case=case_name):
                self.assertTrue(np.all(np.isfinite(a_out)))
                cos_v = float(np.dot(a_out.ravel(), a_ref.ravel()) / (np.linalg.norm(a_out) * np.linalg.norm(a_ref)))
                self.assertGreater(cos_v, 0.99999)

        # 2. DFlash verify dq=4 & dq=5 (d2 split & nod2, SWA + Full, pbf16 + foldcol)
        for dq_v, d2_flag, sw, mode_s in (
            (4, False, 2049, "w8,c1024,noz,pbf16,g4k,foldcol,verify"),
            (4, True, 2049, "w8,c1024,noz,pbf16,g4k,foldcol,verify"),
            (4, True, None, "w8,c1024,noz,pbf16,g4k,foldcol,verify"),
            (5, True, 2049, "w8,c512,noz,pbf16,g4k,foldcol,verify"),
        ):
            kl_np = np.asarray([400, 900, 1400, 1900, 512, 1025, 1537, 2048], dtype=np.int32)
            n_tok = bs * dq_v
            q = jnp.asarray(rng.normal(0.0, 0.4, (n_tok, 16, 128)), jnp.bfloat16)
            kn = jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 60, (n_tok, 1, 128), np.uint8)), jnp.float8_e4m3fn)
            vn = jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 60, (n_tok, 1, 128), np.uint8)), jnp.float8_e4m3fn)
            cuq = jnp.arange(bs + 1, dtype=jnp.int32) * dq_v
            kvc = _make_kvc(kl_np, dq_v, d2_flag)
            out, _ = run_decode_fast_rpa(
                q, kn, vn, kvc, jnp.asarray(kl_np), pi, cuq, cukv, dist,
                sm_scale=128 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                mode_str=mode_s, kv_pack_dense2=d2_flag, decode_query_size=dq_v, interpret=ip,
            )
            a_out = np.asarray(out, dtype=np.float32)
            a_ref = _fp32_ref(q, kn, vn, kl_np, dq_v, sw)
            with self.subTest(dq=dq_v, d2=d2_flag, sw=sw):
                self.assertTrue(np.all(np.isfinite(a_out)))
                cos_v = float(np.dot(a_out.ravel(), a_ref.ravel()) / (np.linalg.norm(a_out) * np.linalg.norm(a_ref)))
                self.assertGreater(cos_v, 0.99999)

        # 2b. TP=16 (8, 64) DFlash verify dq=4 (q8, nod2 + d2 split, 0x7F NaN-poisoned)
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import can_use_decode_fast, parse_decode_fast_config
        for d2_flag in (False, True):
            kl_np = np.asarray([400, 900, 1400, 1900, 512, 1025, 1537, 2048], dtype=np.int32)
            n_tok = bs * 4
            q_tp16 = jnp.asarray(rng.normal(0.0, 0.4, (n_tok, 8, 64)), jnp.bfloat16)
            kn = jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 60, (n_tok, 1, 128), np.uint8)), jnp.float8_e4m3fn)
            vn = jax.lax.bitcast_convert_type(jnp.asarray(rng.integers(0, 60, (n_tok, 1, 128), np.uint8)), jnp.float8_e4m3fn)
            cuq = jnp.arange(bs + 1, dtype=jnp.int32) * 4
            kvc = _make_kvc(kl_np, 4, d2_flag)
            self.assertTrue(
                can_use_decode_fast(
                    mode_str="tp16,w8,c1024,noz,pbf16,g4k,foldcol,noadapt,verify",
                    q=q_tp16,
                    kv_cache_fused=kvc,
                    out_dtype=jnp.bfloat16,
                    decode_query_size=4,
                    custom_mask=None,
                    attention_sink=None,
                    soft_cap=None,
                    xai_temperature_len=None,
                    q_scale=None,
                    actual_num_q_heads=8,
                    actual_num_kv_heads=1,
                    actual_head_dim=64,
                    sliding_window=2049,
                    pages_per_seq=pps,
                    pure_unpacked_decode=True,
                    kv_pack_dense2=d2_flag,
                )
            )
            out, _ = run_decode_fast_rpa(
                q_tp16, kn, vn, kvc, jnp.asarray(kl_np), pi, cuq, cukv, dist,
                sm_scale=64 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                mode_str="tp16,w8,c1024,noz,pbf16,g4k,foldcol,noadapt,verify",
                kv_pack_dense2=d2_flag, decode_query_size=4, force_native_tp16=True, interpret=ip,
            )
            a_out = np.asarray(out, dtype=np.float32)
            # Compute fp32 reference on the 8 Q heads (cols :64) with sm_scale = 64**-0.5
            q_f = np.asarray(q_tp16, dtype=np.float32)
            kn_f = np.asarray(jax.lax.bitcast_convert_type(kn, jnp.float8_e4m3fn).astype(np.float32))[:, :, :64]
            vn_f = np.asarray(jax.lax.bitcast_convert_type(vn, jnp.float8_e4m3fn).astype(np.float32))[:, :, :64]
            refs = []
            for b_i, kl in enumerate(kl_np):
                n_c = int(kl) - 4
                k_c = np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(ru8[b_i, :, :, 0, :]), jnp.float8_e4m3fn).astype(np.float32)).reshape(-1, 128)[:n_c, :64]
                v_c = np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(ru8[b_i, :, :, 1, :]), jnp.float8_e4m3fn).astype(np.float32)).reshape(-1, 128)[:n_c, :64]
                k_seq = np.concatenate([k_c, kn_f[b_i * 4 : (b_i + 1) * 4, 0, :]], axis=0)
                v_seq = np.concatenate([v_c, vn_f[b_i * 4 : (b_i + 1) * 4, 0, :]], axis=0)
                for qi in range(4):
                    tot_k = n_c + qi + 1
                    first_k = max(0, tot_k - 2049)
                    logits = (q_f[b_i * 4 + qi] @ k_seq[first_k:tot_k].T) * (64 ** -0.5)
                    logits -= np.max(logits, axis=-1, keepdims=True)
                    probs = np.exp(logits)
                    probs /= np.sum(probs, axis=-1, keepdims=True)
                    refs.append(np.asarray(jnp.asarray(probs @ v_seq[first_k:tot_k], jnp.bfloat16), dtype=np.float32))
            a_ref = np.stack(refs, axis=0)
            with self.subTest(tp16_dq4_d2=d2_flag):
                self.assertTrue(np.all(np.isfinite(a_out)))
                cos_v = float(np.dot(a_out.ravel(), a_ref.ravel()) / (np.linalg.norm(a_out) * np.linalg.norm(a_ref)))
                self.assertGreater(cos_v, 0.99999)

        # 3. Verify SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY_OPTS default-off, dyn4k/adapt precedence & TPU v7 export
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY_OPTS", None)
            self.assertEqual(envs.SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY_OPTS.get(), "")

        try:
            import torch  # noqa: F401
            from sglang_tpu.attention_pallas_rpa_v3 import _invoke_v3_kernel
            captured = {}
            def _cap(*_a, **kw):
                captured.clear()
                captured.update(kw)
                return "ok"
            with patch.dict(
                os.environ,
                {
                    "SGLANG_TPU_RPA_V3_DECODE_FAST": "w8,c1024,batchfull,dyn4k",
                    "SGLANG_TPU_RPA_V3_DECODE_ADAPTIVE": "1",
                    "SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY": "1",
                    "SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY_OPTS": "auto",
                },
                clear=False,
            ):
                _invoke_v3_kernel(
                    _cap,
                    queries=jax.ShapeDtypeStruct((256, 16, 128), jnp.bfloat16),
                    keys=jax.ShapeDtypeStruct((256, 1, 128), jnp.float8_e4m3fn),
                    values=jax.ShapeDtypeStruct((256, 1, 128), jnp.float8_e4m3fn),
                    kv_cache_fused=jax.ShapeDtypeStruct((256, 512, 1, 4, 128), jnp.float8_e4m3fn),
                    kv_lens=jax.ShapeDtypeStruct((64,), jnp.int32),
                    page_indices=jax.ShapeDtypeStruct((64 * 6,), jnp.int32),
                    cu_q_lens=jax.ShapeDtypeStruct((65,), jnp.int32),
                    cu_kv_lens=jax.ShapeDtypeStruct((65,), jnp.int32),
                    distribution=jax.ShapeDtypeStruct((3,), jnp.int32),
                    attention_sink=None,
                    sm_scale=0.125,
                    sliding_window=2049,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    decode_query_size=4,
                )
                df_dq4 = captured.get("_decode_fast", "")
                self.assertIn("c1024,noz,pbf16,g4k,foldcol,noadapt,verify", df_dq4)
                self.assertNotIn("_adapt", df_dq4)
                cfg_dq4 = parse_decode_fast_config(
                    df_dq4, sliding_window=2049, pages_per_seq=6, page_size=512, decode_query_size=4
                )
                self.assertEqual(cfg_dq4.chunk_sz, 1024)
                self.assertTrue(cfg_dq4.pv_bf16)
                self.assertTrue(cfg_dq4.fold_col)
                _invoke_v3_kernel(
                    _cap,
                    queries=jax.ShapeDtypeStruct((64, 16, 128), jnp.bfloat16),
                    keys=jax.ShapeDtypeStruct((64, 1, 128), jnp.float8_e4m3fn),
                    values=jax.ShapeDtypeStruct((64, 1, 128), jnp.float8_e4m3fn),
                    kv_cache_fused=jax.ShapeDtypeStruct((256, 512, 1, 4, 128), jnp.float8_e4m3fn),
                    kv_lens=jax.ShapeDtypeStruct((64,), jnp.int32),
                    page_indices=jax.ShapeDtypeStruct((64 * 6,), jnp.int32),
                    cu_q_lens=jax.ShapeDtypeStruct((65,), jnp.int32),
                    cu_kv_lens=jax.ShapeDtypeStruct((65,), jnp.int32),
                    distribution=jax.ShapeDtypeStruct((3,), jnp.int32),
                    attention_sink=None,
                    sm_scale=0.125,
                    sliding_window=2049,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    decode_query_size=1,
                )
                self.assertNotIn("pbf16", captured.get("_decode_fast", ""))
                self.assertIn("_adapt", captured.get("_decode_fast", ""))
        except ImportError:
            pass

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            bs_e, dq_e, pps_e = 64, 4, 6
            hlo = jax_export.export(
                jax.jit(
                    lambda q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=128 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                        mode_str="w8,c1024,noz,pbf16,g4k,foldcol,verify",
                        kv_pack_dense2=True, decode_query_size=dq_e,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((bs_e * dq_e, 16, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((bs_e * dq_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e * dq_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((512, 256, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e * pps_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("dq4", hlo)
            self.assertIn("pbf16", hlo)

            hlo_tp16 = jax_export.export(
                jax.jit(
                    lambda q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kn_, vn_, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=64 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                        mode_str="tp16,w8,c1024,noz,pbf16,g4k,foldcol,noadapt,verify",
                        kv_pack_dense2=True, decode_query_size=dq_e, force_native_tp16=True,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((bs_e * dq_e, 8, 64), jnp.bfloat16),
                jax.ShapeDtypeStruct((bs_e * dq_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e * dq_e, 1, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((512, 256, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e * pps_e,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("tp16", hlo_tp16)
            self.assertIn("q8", hlo_tp16)
            self.assertIn("dq4", hlo_tp16)
            self.assertIn("pbf16", hlo_tp16)

    def test_cand12_tp16_q8_shared16_zero_ulp_and_export(self):
        """cand_w9_12: TP=16 q8 in shared16 (LC 256K -38.45 us/occ, 0-ULP vs noq8 & TPU v7 export)."""
        from jax import export as jax_export
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        rng = np.random.default_rng(1208)
        bs, ps, pps = 16, 512, 8
        sh_pg = 3
        lens = [sh_pg * ps + 128] * 12 + [sh_pg * ps + 600, sh_pg * ps + 1, 0, 0]
        n_act = 14
        s_u8 = np.full((bs * pps + 8, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        s_u8[:, :, 0, :2, :] = 0
        s_u8[:, :, 0, 0, :64] = _rand_fp8_u8(rng, (bs * pps + 8, ps, 64))
        s_u8[:, :, 0, 1, :64] = _rand_fp8_u8(rng, (bs * pps + 8, ps, 64))
        pi_np = np.zeros((bs, pps), dtype=np.int32)
        nxt = sh_pg
        for b_i in range(bs):
            pi_np[b_i, :sh_pg] = np.arange(sh_pg, dtype=np.int32)
            pi_np[b_i, sh_pg:] = np.arange(nxt, nxt + (pps - sh_pg), dtype=np.int32)
            nxt += pps - sh_pg
        for s in range(bs):
            if lens[s] <= 0:
                continue
            old_l = lens[s] - 1
            lp = old_l // ps
            rem = old_l % ps
            s_u8[pi_np[s, lp], rem:, 0, :2, :] = 0x7F
            if lp + 1 < pps:
                s_u8[pi_np[s, lp + 1 :], :, 0, :2, :] = 0x7F
        kvc = _u8_to_fp8(s_u8)
        q16 = jnp.asarray((rng.standard_normal((bs, 8, 64)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
        k16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 64)))
        v16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 64)))
        gate = jnp.asarray((rng.standard_normal((bs, 512)) * 0.4).astype(np.float32), dtype=jnp.bfloat16)
        q_prep = jnp.pad(q16, ((0, 0), (0, 8), (0, 64))).reshape(1, bs, 8, 2, 128)
        kv_prep = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs, 1, 4, 128)
        kvl = jnp.asarray(lens, dtype=jnp.int32)
        pi = jnp.asarray(pi_np.ravel(), dtype=jnp.int32)
        cuq = jnp.arange(bs + 1, dtype=jnp.int32)
        cukv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps)
        dist = jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32)
        shp = jnp.asarray([sh_pg], dtype=jnp.int32)

        for sh_tag, ref_m, q8_m in (
            ("shared16", "tp16,noq8,w8,c4096,shared16,q256,pbf16,batchfull,foldcol,noadapt", "tp16,w8,c4096,shared16,q128,pbf16,batchfull,foldcol,noadapt"),
            ("shared8", "tp16,noq8,w8,c4096,shared8,q256,pbf16,batchfull,foldcol,noadapt", "tp16,w8,c4096,shared8,q256,pbf16,batchfull,foldcol,noadapt"),
        ):
            shp_cur = jnp.asarray([sh_pg] * (1 if sh_tag == "shared16" else (bs // 8)), dtype=jnp.int32)
            with self.subTest(shared=sh_tag):
                o_noq8, kv_noq8 = run_decode_fast_rpa(
                    q_prep, kv_prep, None, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=64 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                    mode_str=ref_m, fused_out_pack=2, prepared_inputs=True, gate=gate,
                    out_norm_eps=1e-6, force_native_tp16=True, shared_pages=shp_cur, interpret=True,
                )
                o_q8, kv_q8 = run_decode_fast_rpa(
                    q_prep, kv_prep, None, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=64 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                    mode_str=q8_m, fused_out_pack=2, prepared_inputs=True, gate=gate,
                    out_norm_eps=1e-6, force_native_tp16=True, shared_pages=shp_cur, interpret=True,
                )
                a_ref = np.asarray(o_noq8[:n_act], dtype=np.float32)
                a_q8 = np.asarray(o_q8[:n_act], dtype=np.float32)
                self.assertTrue(np.all(np.isfinite(a_q8)))
                self.assertEqual(float(np.max(np.abs(a_q8 - a_ref))), 0.0)
                self.assertTrue(np.array_equal(np.asarray(kv_q8).view(np.uint8), np.asarray(kv_noq8).view(np.uint8)))

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            pps_256k = 544
            hlo = jax_export.export(
                jax.jit(
                    lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_, sp_: run_decode_fast_rpa(
                        q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=64 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                        mode_str="tp16,w8,c4096,shared16,q256,pbf16,batchfull,foldcol,noadapt",
                        fused_out_pack=2, prepared_inputs=True, gate=g_, out_norm_eps=1e-6,
                        force_native_tp16=True, shared_pages=sp_,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, bs, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((bs, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs, 512), jnp.bfloat16),
                jax.ShapeDtypeStruct((bs * pps_256k + 8, 512, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((bs,), jnp.int32),
                jax.ShapeDtypeStruct((bs * pps_256k,), jnp.int32),
                jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                jax.ShapeDtypeStruct((bs + 1,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            ).mlir_module()
            self.assertIn("shared16", hlo)
            self.assertIn("-tp16", hlo)
            self.assertIn("-q8", hlo)

    def test_cand13_u4mix_adapt2k_and_hoistkv_nan_poisoned_and_export(self):
        """cand_w9_13: mixed u4flat (u4mix), adapt2k Full 1/1K, hoistkv fix, and w7auto c1024 guard."""
        from jax import export as jax_export
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            parse_decode_fast_config,
            run_decode_fast_rpa,
        )

        cfg_w7 = parse_decode_fast_config(
            "w8,c1024,noz,batchfull,u4flat,g4k,foldcol",
            sliding_window=None,
            pages_per_seq=40,
            page_size=512,
        )
        self.assertEqual(cfg_w7.loop_kind, "w7auto")
        self.assertEqual(cfg_w7.chunk_sz, 5120)

        rng = np.random.default_rng(1308)
        bs, ps = 8, 512
        cases = [
            ("tp8_swa_mix_noz", 16, 2049, 6, 8, [100, 513, 640, 1025, 1300, 1450, 1800, 2048], "w8,noz,batchfull,u4flat,g4k,foldcol"),
            ("tp8_swa_mix_nonoz", 16, 2049, 6, 8, [100, 513, 640, 1025, 1300, 1450, 1800, 2048], "w8,nonoz,batchfull,u4flat,g4k,foldcol"),
            ("tp16_swa_mix_noz", 8, 2049, 6, 8, [100, 513, 640, 1025, 1300, 1450, 1800, 2048], "tp16,w8,noz,batchfull,u4flat,g4k,foldcol"),
            ("tp8_swa_unif550", 16, 2049, 6, 8, [550] * 8, "w8,noz,batchfull,u4flat,g4k,foldcol"),
            ("tp8_full_b1025_hkv", 16, None, 12, 8, [100, 250, 513, 700, 800, 900, 1024, 1025], "w8,noz,batchfull,u4flat,hoistkv,g4k,foldcol"),
            ("tp8_full_mix11k_adapt2k_hkv", 16, None, 12, 8, [100, 220, 400, 550, 620, 800, 1020, 1200], "w8,noz,batchfull,u4flat,adapt2k,hoistkv,g4k,foldcol"),
            ("tp16_full_mix11k_adapt2k_nonoz", 8, None, 12, 8, [100, 220, 400, 550, 620, 800, 1020, 1200], "tp16,w8,nonoz,batchfull,u4flat,adapt2k,hoistkv,g4k,foldcol"),
            ("tp8_full_fb_u4_hkv_n3", 16, None, 12, 3, [1, 550, 1200, 1, 1, 1, 1, 1], "w8,noz,batchfull,u4flat,adapt2k,hoistkv,g4k,foldcol"),
            ("tp8_swa_8k1k_mp5_full", 16, 2049, 10, 8, [2100, 2500, 3000, 3500, 4000, 4500, 4800, 5000], "w8,noz,batchfull,u4flat,g4k,foldcol"),
            ("tp8_swa_8k1k_mp5_mix", 16, 2049, 10, 8, [100, 600, 1200, 1800, 2200, 3000, 4000, 5000], "w8,noz,batchfull,u4flat,g4k,foldcol"),
        ]
        for tag, nq, sw, pps, n_act, lens, mode in cases:
            with self.subTest(case=tag):
                s_u8 = np.full((bs * pps + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
                pi_np = np.arange(bs * pps, dtype=np.int32).reshape(bs, pps)
                rng.shuffle(pi_np.ravel())
                cols = 64 if nq == 8 else 128
                for s in range(n_act):
                    old_l = lens[s] - 1
                    n_u = (old_l + ps - 1) // ps
                    for p_o in range(n_u):
                        pg = int(pi_np[s, p_o])
                        rh = min(ps, old_l - p_o * ps)
                        if cols < 128:
                            s_u8[pg, :rh, 0, :2, cols:] = 0
                        s_u8[pg, :rh, 0, 0, :cols] = _rand_fp8_u8(rng, (rh, cols))
                        s_u8[pg, :rh, 0, 1, :cols] = _rand_fp8_u8(rng, (rh, cols))
                kvc = _u8_to_fp8(s_u8)
                q16 = jnp.asarray((rng.standard_normal((bs, 8 if nq == 8 else 16, 64 if nq == 8 else 128)) * 0.4).astype(np.float32), jnp.bfloat16)
                k16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, cols)))
                v16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, cols)))
                if nq == 8:
                    q_p = jnp.pad(q16, ((0, 0), (0, 8), (0, 64))).reshape(1, bs, 8, 2, 128)
                    kv_p = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs, 1, 4, 128)
                else:
                    q_p = q16.reshape(1, bs, 8, 2, 128)
                    kv_p = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 0))).reshape(bs, 1, 4, 128)
                gate = jnp.asarray((rng.standard_normal((bs, nq * 64)) * 0.4).astype(np.float32), jnp.bfloat16)
                pi = jnp.asarray(pi_np.ravel(), jnp.int32)
                cuq = jnp.arange(bs + 1, dtype=jnp.int32)
                cukv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps)
                dist = jnp.asarray([n_act, n_act, n_act], jnp.int32)
                kvl = jnp.asarray(lens, dtype=jnp.int32)
                ref_m = ("tp16," if nq == 8 else "") + "w8,batchfull,g4k,foldcol"
                o_ref, kv_ref = run_decode_fast_rpa(
                    q_p, kv_p, None, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=(64 if nq == 8 else 128) ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                    mode_str=ref_m, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                    gate=gate, out_norm_eps=1e-6, force_native_tp16=(nq == 8), interpret=True,
                )
                o_cand, kv_cand = run_decode_fast_rpa(
                    q_p, kv_p, None, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=(64 if nq == 8 else 128) ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                    mode_str=mode, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                    gate=gate, out_norm_eps=1e-6, force_native_tp16=(nq == 8), interpret=True,
                )
                a_ref = np.asarray(o_ref[:n_act], np.float32)
                a_cand = np.asarray(o_cand[:n_act], np.float32)
                self.assertTrue(np.all(np.isfinite(a_cand)))
                self.assertGreater(_cosine(a_cand, a_ref), 0.99999)
                self.assertTrue(np.array_equal(np.asarray(kv_cand).view(np.uint8), np.asarray(kv_ref).view(np.uint8)))
                if n_act < bs:
                    self.assertTrue(np.all(np.asarray(o_cand[n_act:], np.float32) == 0.0))

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            bs_e, pps_e = 64, 40
            for tag_e, nq_e, sw_e, mode_e in (
                ("tp8_swa_u4mix", 16, 2049, "w8,noz,batchfull,u4flat,g4k,foldcol"),
                ("tp8_full_adapt2k", 16, None, "w8,noz,batchfull,u4flat,adapt2k,g4k,foldcol"),
                ("tp16_full_adapt2k", 8, None, "tp16,w8,noz,batchfull,u4flat,adapt2k,g4k,foldcol"),
            ):
                with self.subTest(export=tag_e):
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                                q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                                sm_scale=(64 if nq_e == 8 else 128) ** -0.5,
                                sliding_window=sw_e, k_scale=1.0, v_scale=1.0,
                                mode_str=mode_e, decode_query_size=1, fused_out_pack=2,
                                prepared_inputs=True, gate=g_, out_norm_eps=1e-6,
                                force_native_tp16=(nq_e == 8),
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, bs_e, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs_e, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs_e, nq_e * 64), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs_e * pps_e + 8, 512, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs_e,), jnp.int32),
                        jax.ShapeDtypeStruct((bs_e * pps_e,), jnp.int32),
                        jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("-u4mix", hlo)
                    if sw_e is None:
                        self.assertIn("swabatch-mp4", hlo)


if __name__ == "__main__":
    unittest.main()
