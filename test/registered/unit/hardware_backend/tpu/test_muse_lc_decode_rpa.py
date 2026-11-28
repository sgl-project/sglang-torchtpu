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

    def test_cand13_hoistkv_u4flat_w7auto_c1024_and_swabatch_nonoz_and_export(self):
        """cand_w9_14: hoistkv fix in u4flat/_fb_u4, w7auto c1024 guard, swabatch nonoz, and kv_len=1025 boundary."""
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

        cfg_swa = parse_decode_fast_config(
            "w8,noz,batchfull,u4flat,hoistkv,g4k,foldcol",
            sliding_window=2049,
            pages_per_seq=40,
            page_size=512,
        )
        self.assertEqual(cfg_swa.loop_kind, "swabatch")
        self.assertFalse(cfg_swa.lazy_slot_zero)

        rng = np.random.default_rng(1308)
        bs, ps = 8, 512
        cases = [
            ("tp8_swa_unif550_u4", 16, 2049, 6, 8, [550] * 8, "w8,noz,batchfull,u4flat,g4k,foldcol"),
            ("tp8_full_b1025_hkv_u4", 16, None, 12, 8, [100, 250, 513, 700, 800, 900, 1024, 1025], "w8,noz,batchfull,u4flat,hoistkv,g4k,foldcol"),
            ("tp8_full_unif900_hkv_u4", 16, None, 12, 8, [900] * 8, "w8,noz,batchfull,u4flat,hoistkv,g4k,foldcol"),
            ("tp16_full_b1025_hkv_u4", 8, None, 12, 8, [100, 250, 513, 700, 800, 900, 1024, 1025], "tp16,w8,noz,batchfull,u4flat,hoistkv,g4k,foldcol"),
            ("tp8_full_fb_u4_hkv_n3", 16, None, 12, 3, [1, 550, 1024, 1, 1, 1, 1, 1], "w8,noz,batchfull,u4flat,hoistkv,g4k,foldcol"),
            ("tp8_swa_8k1k_mp5_full", 16, 2049, 10, 8, [2100, 2500, 3000, 3500, 4000, 4500, 4800, 5000], "w8,noz,batchfull,u4flat,hoistkv,g4k,foldcol"),
            ("tp8_swa_8k1k_mp5_mix", 16, 2049, 10, 8, [100, 600, 1200, 1800, 2200, 3000, 4000, 5000], "w8,noz,batchfull,u4flat,hoistkv,g4k,foldcol"),
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
                ("tp8_swa_nonoz", 16, 2049, "w8,noz,batchfull,u4flat,g4k,foldcol"),
                ("tp8_full_hkv_c5120", 16, None, "w8,c1024,noz,batchfull,u4flat,hoistkv,g4k,foldcol"),
                ("tp16_full_hkv_c5120", 8, None, "tp16,w8,c1024,noz,batchfull,u4flat,hoistkv,g4k,foldcol"),
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
                    self.assertIn("swabatch-mp2-batchfull", hlo)

    def test_cand15_rpad_vmem_core_and_prestarted_dma_parity_and_export(self):
        """cand_w9_15: _run_rpad_vmem_core + prepare_and_start_rpad_dmas bit-exact parity & AOT."""
        from jax import export as jax_export
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            _run_rpad_vmem_core,
            parse_decode_fast_config,
            prepare_and_start_rpad_dmas,
            run_decode_fast_rpa,
        )

        cfg_sh = parse_decode_fast_config(
            "w8,c4096,shared16,q256,pbf16,batchfull,foldcol,noadapt",
            sliding_window=None,
            pages_per_seq=528,
            page_size=512,
            decode_query_size=1,
            shared_group=16,
        )
        self.assertEqual(cfg_sh.num_slots, 4)

        def _run_fused_caller(
            q_raw,
            kv_merged,
            gate_in,
            kvc,
            kvl,
            pi,
            cuq,
            cukv,
            dist,
            *,
            sm_scale: float,
            sw: int | None,
            mode_str: str,
            native_tp16: bool,
            interpret: bool,
        ):
            bs = q_raw.shape[0]
            pps = pi.shape[0] // bs
            cfg = parse_decode_fast_config(
                mode_str,
                sliding_window=sw,
                pages_per_seq=pps,
                page_size=512,
                decode_query_size=1,
            )
            use_tp16_q8 = native_tp16
            q_rows = 8 if use_tp16_q8 else 16
            pack2_cols = 512 if native_tp16 else 1024
            one_block = True if sw is not None else (pps <= cfg.max_ring_pages)
            use_hoist = cfg.hoist_new_kv

            def _kern(
                kv_lens_ref,
                page_indices_ref,
                cu_q_lens_ref,
                cu_kv_lens_ref,
                distribution_ref,
                q_in_ref,
                kv_in_ref,
                gate_in_ref,
                kv_cache_hbm_ref,
                o_out_ref,
                upd_kv_hbm_ref,
                bkv_vmem_ref,
                state_vmem_ref,
                o_vmem_ref,
                *extra_refs,
            ):
                if use_hoist:
                    init_sv_ref, smem_ref, bkv_sems, cw_sem = extra_refs
                else:
                    init_sv_ref = None
                    smem_ref, bkv_sems, cw_sem = extra_refs
                pre_state = prepare_and_start_rpad_dmas(
                    kv_lens_ref=kv_lens_ref,
                    page_indices_ref=page_indices_ref,
                    cu_q_lens_ref=cu_q_lens_ref,
                    cu_kv_lens_ref=cu_kv_lens_ref,
                    distribution_ref=distribution_ref,
                    kv_cache_hbm_ref=kv_cache_hbm_ref,
                    bkv_vmem_ref=bkv_vmem_ref,
                    bkv_sems=bkv_sems,
                    sliding_window=sw,
                    cfg=cfg,
                    one_block_per_seq=one_block,
                    decode_query_size=1,
                    kv_pack_dense2=False,
                    interpret_mode=interpret,
                )
                q_ref = q_in_ref.at[:, pl.ds(0, 8), :] if use_tp16_q8 else q_in_ref
                _run_rpad_vmem_core(
                    kv_lens_ref=kv_lens_ref,
                    page_indices_ref=page_indices_ref,
                    cu_q_lens_ref=cu_q_lens_ref,
                    cu_kv_lens_ref=cu_kv_lens_ref,
                    distribution_ref=distribution_ref,
                    q_vmem_ref=q_ref,
                    kv_merged_vmem_ref=kv_in_ref,
                    gate_vmem_ref=gate_in_ref,
                    kv_cache_hbm_ref=kv_cache_hbm_ref,
                    o_out_vmem_ref=o_out_ref,
                    updated_kv_cache_hbm_ref=upd_kv_hbm_ref,
                    bkv_vmem_ref=bkv_vmem_ref,
                    state_vmem_ref=state_vmem_ref,
                    o_vmem_ref=o_vmem_ref,
                    kv_write_vmem_ref=None,
                    init_sv_vmem_ref=init_sv_ref,
                    written_cnt_smem_ref=smem_ref,
                    bkv_sems=bkv_sems,
                    cache_write_sem=cw_sem,
                    sm_scale=sm_scale,
                    sliding_window=sw,
                    k_scale=1.0,
                    v_scale=1.0,
                    cfg=cfg,
                    one_block_per_seq=one_block,
                    decode_query_size=1,
                    in_kernel_pack2=True,
                    has_gate=True,
                    out_norm_eps=1e-5,
                    kv_pack_dense2=False,
                    interpret_mode=interpret,
                    prestarted_bkv_state=pre_state,
                )

            scratch = [
                pltpu.VMEM((cfg.num_slots, cfg.max_ring_pages, 512, 1, 4, 128), kvc.dtype),
                pltpu.VMEM((3, q_rows, 128), jnp.float32),
                pltpu.VMEM((bs, q_rows, 128), jnp.bfloat16),
            ]
            if use_hoist:
                scratch.append(pltpu.VMEM((2, bs, q_rows, 128), jnp.float32))
            scratch.extend([
                pltpu.SMEM((2 + cfg.num_slots,), jnp.int32),
                pltpu.SemaphoreType.DMA((cfg.num_slots,)),
                pltpu.SemaphoreType.DMA((1,)),
            ])
            tpu_v = 0 if interpret else 7
            kw = (
                {"interpret": pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")}
                if interpret
                else {"compiler_params": pltpu.CompilerParams(dimension_semantics=("arbitrary",), disable_bounds_checks=True)}
            )
            return pl.pallas_call(
                _kern,
                out_shape=[
                    jax.ShapeDtypeStruct((bs, pack2_cols), jnp.bfloat16),
                    pltpu.HBM(shape=kvc.shape, dtype=kvc.dtype) if tpu_v >= 7 else jax.ShapeDtypeStruct(shape=kvc.shape, dtype=kvc.dtype),
                ],
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=5,
                    in_specs=[
                        pl.BlockSpec(q_raw.shape, lambda *_: (0, 0, 0), memory_space=pltpu.VMEM),
                        pl.BlockSpec(kv_merged.shape, lambda *_: (0, 0, 0, 0), memory_space=pltpu.VMEM),
                        pl.BlockSpec(gate_in.shape, lambda *_: (0, 0), memory_space=pltpu.VMEM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                    ],
                    out_specs=[
                        pl.BlockSpec((bs, pack2_cols), lambda *_: (0, 0), memory_space=pltpu.VMEM),
                        pl.BlockSpec(memory_space=pltpu.HBM),
                    ],
                    grid=(1,),
                    scratch_shapes=scratch,
                ),
                input_output_aliases={8: 1},
                name=f"fused_c{cfg.chunk_sz}-{cfg.loop_kind}",
                **kw,
            )(kvl, pi, cuq, cukv, dist, q_raw, kv_merged, gate_in, kvc)

        rng = np.random.default_rng(1501)
        bs, ps = 32, 512
        cases = [
            ("tp8_1k_swa_mix_n28", 16, 2049, 2, 28, [120, 480, 513, 900, 1024] * 5 + [100, 600, 950] + [1] * 4, "w8,batchfull,g4k,foldcol"),
            ("tp16_q8_1k_full_mix", 8, None, 2, 32, [64, 300, 512, 750] * 8, "tp16,w8,batchfull,g4k,foldcol"),
            ("tp8_8k1k_swa_mp5", 16, 2049, 10, 30, [400, 1100, 1800, 2200, 3500, 4500] * 5 + [1, 1], "w8,noadapt,hoistkv,g4k,foldcol"),
            ("tp8_8k1k_full_w7auto", 16, None, 10, 32, [100, 700, 1500, 2600] * 8, "w8,noadapt,hoistkv,g4k,foldcol"),
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
                q16 = jnp.asarray((rng.standard_normal((bs, 8 if nq == 8 else 16, cols)) * 0.4).astype(np.float32), jnp.bfloat16)
                k16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, cols)))
                v16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, cols)))
                if nq == 8:
                    q_raw = jnp.pad(q16, ((0, 0), (0, 8), (0, 64)))
                    kv_p = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs, 1, 4, 128)
                else:
                    q_raw = q16
                    kv_p = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 0))).reshape(bs, 1, 4, 128)
                gate = jnp.asarray((rng.standard_normal((bs, nq * 64)) * 0.4).astype(np.float32), jnp.bfloat16)
                pi = jnp.asarray(pi_np.ravel(), jnp.int32)
                cuq = jnp.arange(bs + 1, dtype=jnp.int32)
                cukv = jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps)
                dist = jnp.asarray([n_act, n_act, n_act], jnp.int32)
                kvl = jnp.asarray(lens, dtype=jnp.int32)

                o_ref, kv_ref = run_decode_fast_rpa(
                    q_raw.reshape(1, bs, 8, 2, 128), kv_p, None, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=float(cols ** -0.5), sliding_window=sw, k_scale=1.0, v_scale=1.0,
                    mode_str=mode, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                    gate=gate, out_norm_eps=1e-5, force_native_tp16=(nq == 8), interpret=True,
                )
                o_fused, kv_fused = _run_fused_caller(
                    q_raw, kv_p, gate, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=float(cols ** -0.5), sw=sw, mode_str=mode, native_tp16=(nq == 8), interpret=True,
                )
                self.assertTrue(np.array_equal(np.asarray(o_fused).view(np.uint16), np.asarray(o_ref).view(np.uint16)))
                self.assertTrue(np.array_equal(np.asarray(kv_fused).view(np.uint8), np.asarray(kv_ref).view(np.uint8)))

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            bs_e = 64
            for tag_e, nq_e, sw_e, pps_e, mode_e in (
                ("tp8_swa_1k", 16, 2049, 2, "w8,batchfull,g4k,foldcol"),
                ("tp16_q8_swa_8k1k", 8, 2049, 5, "tp16,w8,noadapt,hoistkv,g4k,foldcol"),
                ("tp8_full_8k1k", 16, None, 17, "w8,noadapt,hoistkv,g4k,foldcol"),
            ):
                with self.subTest(export=tag_e):
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: _run_fused_caller(
                                q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_,
                                sm_scale=(64.0 if nq_e == 8 else 128.0) ** -0.5,
                                sw=sw_e, mode_str=mode_e, native_tp16=(nq_e == 8), interpret=False,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((bs_e, 16, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs_e, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs_e, nq_e * 64), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs_e * pps_e + 8, 512, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs_e,), jnp.int32),
                        jax.ShapeDtypeStruct((bs_e * pps_e,), jnp.int32),
                        jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("tpu_custom_call", hlo.lower())
                    self.assertNotIn("swabatch-mp5-noz", hlo)
                    if sw_e is None:
                        self.assertIn("c5120-w7auto", hlo)

    def test_cand16_batch4_swabatch_parity_prestart_and_export(self):
        """cand_w9_16: vectorized batch4/batch2 swabatch, prestarted DMA hook, v_scale, fallback, and AOT."""
        from jax import export as jax_export
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            _run_rpad_vmem_core,
            parse_decode_fast_config,
            prepare_and_start_rpad_dmas,
            run_decode_fast_rpa,
        )

        cfg_b4 = parse_decode_fast_config(
            "w8,batchfull,g4k,foldcol,batch4",
            sliding_window=2049,
            pages_per_seq=4,
            page_size=512,
        )
        self.assertEqual(cfg_b4.swabatch_group_size, 4)
        cfg_w7 = parse_decode_fast_config(
            "w8,noadapt,hoistkv,g4k,foldcol,batch4",
            sliding_window=None,
            pages_per_seq=17,
            page_size=512,
        )
        self.assertEqual(cfg_w7.swabatch_group_size, 1)

        rng = np.random.default_rng(1602)
        ps = 512
        cases = [
            ("tp8_1k_swa_b4_n29_vsc", 64, 16, 2049, 4, 29, [120, 480, 513, 900, 1024] * 5 + [1, 300, 600, 950] + [1] * 35, "w8,batchfull,g4k,foldcol,batch4", 0.75, True),
            ("tp16_q8_1k_full_b4", 32, 8, None, 2, 30, [64, 300, 512, 750] * 7 + [200, 880] + [1, 1], "tp16,w8,batchfull,g4k,foldcol,batch4", 1.25, True),
            ("tp8_8k1k_swa_b4_s3", 32, 16, 2049, 10, 31, [400, 1100, 1800, 2200, 3500, 4500] * 5 + [2700, 1], "w8,s3,noadapt,g4k,foldcol,batch4", 1.0, True),
            ("tp8_b32_8k1k_swa_b2", 32, 16, 2049, 10, 30, [500, 1400, 2600, 3900] * 7 + [100, 1500, 3100, 1], "w8,noadapt,g4k,foldcol,batch2", 1.0, True),
            ("tp8_1k_nobatch4_noncontig", 32, 16, 2049, 2, 24, [200, 600, 900] * 8 + [1] * 8, "w8,batchfull,g4k,foldcol,batch4,nobatch4", 1.0, False),
        ]
        for tag, bs, nq, sw, pps, n_act, lens, mode, v_sc, contig_cuq in cases:
            with self.subTest(case=tag):
                s_u8 = np.full((bs * pps + 8, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
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
                q16 = jnp.asarray((rng.standard_normal((bs, 8 if nq == 8 else 16, cols)) * 0.4).astype(np.float32), jnp.bfloat16)
                k16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, cols)))
                v16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, cols)))
                if nq == 8:
                    q_raw = jnp.pad(q16, ((0, 0), (0, 8), (0, 64)))
                    kv_p = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs, 1, 4, 128)
                else:
                    q_raw = q16
                    kv_p = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 0))).reshape(bs, 1, 4, 128)
                gate = jnp.asarray((rng.standard_normal((bs, nq * 64)) * 0.4).astype(np.float32), jnp.bfloat16)
                pi = jnp.asarray(pi_np.ravel(), jnp.int32)
                if contig_cuq:
                    cuq_np = np.arange(bs + 1, dtype=np.int32)
                else:
                    # One skipped zero-qlen entry in the middle triggers cu_q_lens_ref[n_active] != n_active fallback
                    cuq_np = np.arange(bs + 1, dtype=np.int32)
                    cuq_np[3:] -= 1
                    cuq_np[-1] = cuq_np[-2]
                cuq = jnp.asarray(cuq_np, jnp.int32)
                cukv = jnp.arange(bs + 1, dtype=np.int32) * (pps * ps)
                dist = jnp.asarray([n_act, n_act, n_act], jnp.int32)
                kvl = jnp.asarray(lens, dtype=jnp.int32)

                ref_mode = mode.replace(",batch4", "").replace(",batch2", "")
                o_ref, kv_ref = run_decode_fast_rpa(
                    q_raw.reshape(1, bs, 8, 2, 128), kv_p, None, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=float(cols ** -0.5), sliding_window=sw, k_scale=1.0, v_scale=v_sc,
                    mode_str=ref_mode, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                    gate=gate, out_norm_eps=1e-5, force_native_tp16=(nq == 8), interpret=True,
                )
                o_b4, kv_b4 = run_decode_fast_rpa(
                    q_raw.reshape(1, bs, 8, 2, 128), kv_p, None, kvc, kvl, pi, cuq, cukv, dist,
                    sm_scale=float(cols ** -0.5), sliding_window=sw, k_scale=1.0, v_scale=v_sc,
                    mode_str=mode, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                    gate=gate, out_norm_eps=1e-5, force_native_tp16=(nq == 8), interpret=True,
                )
                act_toks = int(cuq_np[n_act])
                a_r = np.asarray(o_ref[:act_toks], np.float32)
                a_4 = np.asarray(o_b4[:act_toks], np.float32)
                self.assertTrue(np.all(np.isfinite(np.asarray(o_b4, np.float32))))
                self.assertGreater(_cosine(a_4, a_r), 0.99999)
                self.assertLessEqual(float(np.max(np.abs(a_4 - a_r))), 1e-2)
                self.assertTrue(np.all(np.asarray(o_b4[act_toks:], np.float32) == 0.0))
                self.assertTrue(np.array_equal(np.asarray(kv_b4).view(np.uint8), np.asarray(kv_ref).view(np.uint8)))

        # Prestarted DMA hook parity with batch4 (Step c + b4_s3 6 MiB config)
        bs, nq, sw, pps, n_act = 32, 16, 2049, 4, 29
        mode_pre = "w8,s3,batchfull,g4k,foldcol,batch4"
        cfg_pre = parse_decode_fast_config(mode_pre, sliding_window=sw, pages_per_seq=pps, page_size=ps, decode_query_size=1)
        lens_p = [120, 480, 513, 900, 1024] * 5 + [1, 300, 600, 950] + [1, 1, 1]
        s_u8 = np.full((bs * pps + 8, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        pi_np = np.arange(bs * pps, dtype=np.int32).reshape(bs, pps)
        rng.shuffle(pi_np.ravel())
        for s in range(n_act):
            old_l = lens_p[s] - 1
            for p_o in range((old_l + ps - 1) // ps):
                pg = int(pi_np[s, p_o])
                rh = min(ps, old_l - p_o * ps)
                s_u8[pg, :rh, 0, 0, :] = _rand_fp8_u8(rng, (rh, 128))
                s_u8[pg, :rh, 0, 1, :] = _rand_fp8_u8(rng, (rh, 128))
        kvc = _u8_to_fp8(s_u8)
        q_raw = jnp.asarray((rng.standard_normal((bs, 16, 128)) * 0.4).astype(np.float32), jnp.bfloat16)
        k16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 128)))
        v16 = _u8_to_fp8(_rand_fp8_u8(rng, (bs, 1, 128)))
        kv_p = jnp.pad(jnp.concatenate([k16, v16], axis=1), ((0, 0), (0, 2), (0, 0))).reshape(bs, 1, 4, 128)
        gate = jnp.asarray((rng.standard_normal((bs, 1024)) * 0.4).astype(np.float32), jnp.bfloat16)
        pi = jnp.asarray(pi_np.ravel(), jnp.int32)
        cuq = jnp.arange(bs + 1, dtype=jnp.int32)
        cukv = jnp.arange(bs + 1, dtype=np.int32) * (pps * ps)
        dist = jnp.asarray([n_act, n_act, n_act], jnp.int32)
        kvl = jnp.asarray(lens_p, dtype=jnp.int32)

        o_std, kv_std = run_decode_fast_rpa(
            q_raw.reshape(1, bs, 8, 2, 128), kv_p, None, kvc, kvl, pi, cuq, cukv, dist,
            sm_scale=128.0 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
            mode_str=mode_pre, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
            gate=gate, out_norm_eps=1e-5, interpret=True,
        )

        def _pre_kern(klr, pir, cqr, ckvr, dr, q_r, kv_r, g_r, chbm, oout, upd, bkv_r, st_r, o_r, init_r, sm_r, s1, s2):
            pre_st = prepare_and_start_rpad_dmas(
                kv_lens_ref=klr, page_indices_ref=pir, cu_q_lens_ref=cqr, cu_kv_lens_ref=ckvr,
                distribution_ref=dr, kv_cache_hbm_ref=chbm, bkv_vmem_ref=bkv_r, bkv_sems=s1,
                sliding_window=sw, cfg=cfg_pre, one_block_per_seq=True, decode_query_size=1,
                kv_pack_dense2=False, interpret_mode=True,
            )
            _run_rpad_vmem_core(
                kv_lens_ref=klr, page_indices_ref=pir, cu_q_lens_ref=cqr, cu_kv_lens_ref=ckvr,
                distribution_ref=dr, q_vmem_ref=q_r, kv_merged_vmem_ref=kv_r, gate_vmem_ref=g_r,
                kv_cache_hbm_ref=chbm, o_out_vmem_ref=oout, updated_kv_cache_hbm_ref=upd,
                bkv_vmem_ref=bkv_r, state_vmem_ref=st_r, o_vmem_ref=o_r, kv_write_vmem_ref=None,
                init_sv_vmem_ref=init_r, written_cnt_smem_ref=sm_r, bkv_sems=s1, cache_write_sem=s2,
                sm_scale=128.0 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0, cfg=cfg_pre,
                one_block_per_seq=True, decode_query_size=1, in_kernel_pack2=True, has_gate=True,
                out_norm_eps=1e-5, kv_pack_dense2=False, interpret_mode=True, prestarted_bkv_state=pre_st,
            )

        o_pre, kv_pre = pl.pallas_call(
            _pre_kern,
            out_shape=[jax.ShapeDtypeStruct((bs, 1024), jnp.bfloat16), jax.ShapeDtypeStruct(kvc.shape, kvc.dtype)],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=5,
                in_specs=[
                    pl.BlockSpec(q_raw.shape, lambda *_: (0, 0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(kv_p.shape, lambda *_: (0, 0, 0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(gate.shape, lambda *_: (0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=[
                    pl.BlockSpec((bs, 1024), lambda *_: (0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM((cfg_pre.num_slots, 4 * cfg_pre.max_ring_pages, 512, 1, 4, 128), kvc.dtype),
                    pltpu.VMEM((3, 16, 128), jnp.float32),
                    pltpu.VMEM((bs, 16, 128), jnp.bfloat16),
                    pltpu.VMEM((2, bs, 16, 128), jnp.float32),
                    pltpu.SMEM((2 + cfg_pre.num_slots,), jnp.int32),
                    pltpu.SemaphoreType.DMA((cfg_pre.num_slots,)),
                    pltpu.SemaphoreType.DMA((1,)),
                ],
            ),
            input_output_aliases={8: 1},
            interpret=pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan"),
        )(kvl, pi, cuq, cukv, dist, q_raw, kv_p, gate, kvc)
        self.assertTrue(np.array_equal(np.asarray(o_pre).view(np.uint16), np.asarray(o_std).view(np.uint16)))
        self.assertTrue(np.array_equal(np.asarray(kv_pre).view(np.uint8), np.asarray(kv_std).view(np.uint8)))

        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            for tag_e, bs_e, nq_e, sw_e, pps_e, mode_e in (
                ("tp8_1k_b4_s4", 64, 16, 2049, 2, "w8,batchfull,g4k,foldcol,batch4"),
                ("tp8_1k_adapt_pps4_b4", 64, 16, 2049, 4, "w8,batchfull,g4k,foldcol,batch4"),
                ("tp16_q8_1k_b4_s4", 64, 8, None, 2, "tp16,w8,batchfull,g4k,foldcol,batch4"),
                ("tp8_8k1k_swa_b4_s3", 64, 16, 2049, 5, "w8,s3,noadapt,g4k,foldcol,batch4"),
                ("tp16_b32_8k1k_swa_b2", 32, 8, 2049, 5, "tp16,w8,noadapt,g4k,foldcol,batch2"),
            ):
                with self.subTest(export=tag_e):
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                                q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                                sm_scale=(64.0 if nq_e == 8 else 128.0) ** -0.5,
                                sliding_window=sw_e, k_scale=1.0, v_scale=1.0,
                                mode_str=mode_e, decode_query_size=1, fused_out_pack=2,
                                prepared_inputs=True, gate=g_, out_norm_eps=1e-5,
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
                    self.assertIn("tpu_custom_call", hlo.lower())
                    self.assertIn("-b2" if "batch2" in mode_e else "-b4", hlo)
                    if tag_e == "tp8_1k_adapt_pps4_b4":
                        self.assertIn("RPAd_fast-s4-c2048-swabatch-mp5", hlo)
                        self.assertNotIn("c1024-swabatch-mp2", hlo)

    def test_cand16_swabatch_group_predicate_alignment_and_adaptive_c1024_gate(self):
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            _run_rpad_vmem_core,
            can_use_swabatch_group_runtime,
            parse_decode_fast_config,
            prepare_and_start_rpad_dmas,
            run_decode_fast_rpa,
        )

        cfg = parse_decode_fast_config(
            "w8,s4,batchfull,g4k,foldcol,batch4",
            sliding_window=2049,
            pages_per_seq=4,
            page_size=512,
            decode_query_size=1,
        )
        self.assertTrue(
            can_use_swabatch_group_runtime(
                cfg,
                one_block_per_seq=True,
                decode_query_size=1,
                kv_pack_dense2=False,
                probe=None,
                b_tokens=64,
                max_num_seqs=64,
                bkv_slot_pages=cfg.swabatch_group_size * cfg.max_ring_pages,
                has_init_sv=True,
            )
        )
        # Missing init_sv or undersized bkv or mismatched b_tokens must return False.
        self.assertFalse(
            can_use_swabatch_group_runtime(
                cfg,
                one_block_per_seq=True,
                decode_query_size=1,
                kv_pack_dense2=False,
                probe=None,
                b_tokens=64,
                max_num_seqs=64,
                bkv_slot_pages=cfg.swabatch_group_size * cfg.max_ring_pages,
                has_init_sv=False,
            )
        )
        self.assertFalse(
            can_use_swabatch_group_runtime(
                cfg,
                one_block_per_seq=True,
                decode_query_size=1,
                kv_pack_dense2=False,
                probe=None,
                b_tokens=64,
                max_num_seqs=64,
                bkv_slot_pages=cfg.max_ring_pages,
                has_init_sv=True,
            )
        )
        self.assertFalse(
            can_use_swabatch_group_runtime(
                cfg,
                one_block_per_seq=True,
                decode_query_size=1,
                kv_pack_dense2=False,
                probe=None,
                b_tokens=32,
                max_num_seqs=64,
                bkv_slot_pages=cfg.swabatch_group_size * cfg.max_ring_pages,
                has_init_sv=True,
            )
        )

        # Explicit use_grp_vec=True with missing init_sv or undersized bkv must fail fast.
        bs, sw, pps, ps = 8, 2049, 4, 512
        cfg_s = parse_decode_fast_config(
            "w8,s4,batchfull,g4k,foldcol,batch4",
            sliding_window=sw,
            pages_per_seq=pps,
            page_size=ps,
            decode_query_size=1,
        )
        v_sp, h_sp = pl.BlockSpec(memory_space=pltpu.VMEM), pl.BlockSpec(memory_space=pltpu.HBM)

        def _bad_prestart(klr, pir, cqr, ckvr, dr, chbm, out, bkv_r, bkv_s):
            prepare_and_start_rpad_dmas(
                kv_lens_ref=klr, page_indices_ref=pir, cu_q_lens_ref=cqr, cu_kv_lens_ref=ckvr,
                distribution_ref=dr, kv_cache_hbm_ref=chbm, bkv_vmem_ref=bkv_r, bkv_sems=bkv_s,
                sliding_window=sw, cfg=cfg_s, one_block_per_seq=True, decode_query_size=1,
                kv_pack_dense2=False, interpret_mode=True, b_tokens=bs, init_sv_vmem_ref=None,
                use_grp_vec=True,
            )

        with self.assertRaisesRegex(ValueError, "use_grp_vec=True"):
            pl.pallas_call(
                _bad_prestart,
                out_shape=jax.ShapeDtypeStruct((bs, 16, 128), jnp.bfloat16),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=5,
                    in_specs=(h_sp,),
                    out_specs=v_sp,
                    grid=(1,),
                    scratch_shapes=(
                        pltpu.VMEM((cfg_s.num_slots, 4 * cfg_s.max_ring_pages, ps, 1, 4, 128), jnp.float8_e4m3fn),
                        pltpu.SemaphoreType.DMA((cfg_s.num_slots,)),
                    ),
                ),
                interpret=True,
            )(
                jnp.ones((bs,), jnp.int32),
                jnp.arange(bs * pps, dtype=jnp.int32),
                jnp.arange(bs + 1, dtype=jnp.int32),
                jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps),
                jnp.asarray([bs, bs, bs], jnp.int32),
                jnp.zeros((bs * pps, ps, 1, 4, 128), jnp.float8_e4m3fn),
            )

        def _bad_core(klr, pir, cqr, ckvr, dr, q_r, kv_r, chbm, oout, upd, bkv_r, st_r, o_r, sm_r, s1, s2):
            _run_rpad_vmem_core(
                kv_lens_ref=klr, page_indices_ref=pir, cu_q_lens_ref=cqr, cu_kv_lens_ref=ckvr,
                distribution_ref=dr, q_vmem_ref=q_r, kv_merged_vmem_ref=kv_r, gate_vmem_ref=None,
                kv_cache_hbm_ref=chbm, o_out_vmem_ref=oout, updated_kv_cache_hbm_ref=upd,
                bkv_vmem_ref=bkv_r, state_vmem_ref=st_r, o_vmem_ref=o_r, kv_write_vmem_ref=None,
                init_sv_vmem_ref=None, written_cnt_smem_ref=sm_r, bkv_sems=s1, cache_write_sem=s2,
                sm_scale=128.0 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0, cfg=cfg_s,
                one_block_per_seq=True, decode_query_size=1, in_kernel_pack2=True, has_gate=False,
                kv_pack_dense2=False, interpret_mode=True, use_grp_vec=True,
            )

        with self.assertRaisesRegex(ValueError, "use_grp_vec=True"):
            pl.pallas_call(
                _bad_core,
                out_shape=(
                    jax.ShapeDtypeStruct((bs, 16, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((bs * pps, ps, 1, 4, 128), jnp.float8_e4m3fn),
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=5,
                    in_specs=(v_sp, v_sp, h_sp),
                    out_specs=(v_sp, h_sp),
                    grid=(1,),
                    scratch_shapes=(
                        pltpu.VMEM((cfg_s.num_slots, cfg_s.max_ring_pages, ps, 1, 4, 128), jnp.float8_e4m3fn),
                        pltpu.VMEM((3, 16, 128), jnp.float32),
                        pltpu.VMEM((bs, 16, 128), jnp.bfloat16),
                        pltpu.SMEM((2 + cfg_s.num_slots,), jnp.int32),
                        pltpu.SemaphoreType.DMA((cfg_s.num_slots,)),
                        pltpu.SemaphoreType.DMA((1,)),
                    ),
                ),
                input_output_aliases={7: 1},
                interpret=True,
            )(
                jnp.ones((bs,), jnp.int32),
                jnp.arange(bs * pps, dtype=jnp.int32),
                jnp.arange(bs + 1, dtype=jnp.int32),
                jnp.arange(bs + 1, dtype=jnp.int32) * (pps * ps),
                jnp.asarray([bs, bs, bs], jnp.int32),
                jnp.zeros((bs, 16, 128), jnp.bfloat16),
                jnp.zeros((bs, 1, 4, 128), jnp.float8_e4m3fn),
                jnp.zeros((bs * pps, ps, 1, 4, 128), jnp.float8_e4m3fn),
            )

        # Verify use_adaptive_c1024 only bypasses lax.switch when _uses_swabatch_group is True
        # at runtime (not just cfg.swabatch_group_size > 1 with dense2/mismatched seqs).
        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            return
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            hlo_adapt = jax_export.export(
                jax.jit(
                    lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=128.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                        mode_str="w8,s4,batchfull,g4k,foldcol,batch4,dense2,nod2grp",
                        decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                        gate=g_, out_norm_eps=1e-5, kv_pack_dense2=True,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, 64, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((64, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64, 1024), jnp.bfloat16),
                jax.ShapeDtypeStruct((264, 256, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64,), jnp.int32),
                jax.ShapeDtypeStruct((256,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("stablehlo.case", hlo_adapt)
            hlo_d2grp = jax_export.export(
                jax.jit(
                    lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=128.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                        mode_str="w8,s4,batchfull,g4k,foldcol",
                        decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                        gate=g_, out_norm_eps=1e-5, kv_pack_dense2=True,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, 64, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((64, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64, 1024), jnp.bfloat16),
                jax.ShapeDtypeStruct((264, 256, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64,), jnp.int32),
                jax.ShapeDtypeStruct((256,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertNotIn("stablehlo.case", hlo_d2grp)
            hlo_direct = jax_export.export(
                jax.jit(
                    lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=128.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                        mode_str="w8,s4,batchfull,g4k,foldcol,batch4",
                        decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                        gate=g_, out_norm_eps=1e-5, kv_pack_dense2=False,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, 64, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((64, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64, 1024), jnp.bfloat16),
                jax.ShapeDtypeStruct((264, 512, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((64,), jnp.int32),
                jax.ShapeDtypeStruct((256,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((65,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertNotIn("stablehlo.case", hlo_direct)

    def test_cand17_tp16_bd4_full_chain_parity_and_export(self):
        """Verify TP=16 bd4_128B (4x KV pack) pool, RMW cache writes, RPAm, RPAd_fast (b4/b2/w7_pps17/LC/StepC), and v7 AOT export."""
        import os
        from unittest.mock import patch

        import jax
        from jax import export as jax_export
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        import numpy as np
        import torch

        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
            write_kv_only_rpa_v3,
        )
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            _run_rpad_vmem_core,
            parse_decode_fast_config,
            prepare_and_start_rpad_dmas,
            run_decode_fast_rpa,
        )
        from sglang_tpu.attention_pallas_rpa_v3 import (
            RPA_V3_SPEC,
            _get_write_kv_only_op,
            _pool_kv_pack_bd4,
        )
        from sglang_tpu.envs import envs
        from sglang_tpu.kv_pool import (
            TpuDraftFusedMHATokenToKVPool,
            TpuFusedMHATokenToKVPool,
        )

        # 1. Pool metadata, default-OFF guard, TP=8 guard, _logical_kv_view round-trip, & _invoke_v3_kernel
        self.assertFalse(envs.SGLANG_TPU_KV_PACK_BD4.get())
        pool_off = TpuFusedMHATokenToKVPool(
            size=1024, page_size=512, dtype=torch.float8_e4m3fn,
            head_num=1, head_dim=64, layer_num=1, device="cpu", enable_memory_saver=False,
        )
        self.assertFalse(pool_off.kv_pack_bd4)
        self.assertFalse(_pool_kv_pack_bd4(pool_off))
        self.assertEqual(pool_off.kv_buffer[0].shape, (3, 512, 1, 4, 128))
        self.assertEqual(
            TpuFusedMHATokenToKVPool.kv_bytes_per_token_per_layer(
                num_kv_heads=1, head_dim=64, v_head_dim=64, store_dtype=torch.float8_e4m3fn,
            ),
            512,
        )

        with patch.dict(os.environ, {"SGLANG_TPU_KV_PACK_BD4": "1"}):
            self.assertEqual(
                TpuFusedMHATokenToKVPool.kv_bytes_per_token_per_layer(
                    num_kv_heads=1, head_dim=64, v_head_dim=64, store_dtype=torch.float8_e4m3fn,
                ),
                128,
            )
            self.assertEqual(
                TpuFusedMHATokenToKVPool.kv_bytes_per_token_per_layer(
                    num_kv_heads=2, head_dim=64, v_head_dim=64, store_dtype=torch.float8_e4m3fn,
                ),
                512,
            )
            pool_bd4 = TpuFusedMHATokenToKVPool(
                size=1024, page_size=512, dtype=torch.float8_e4m3fn,
                head_num=1, head_dim=64, layer_num=1, device="cpu", enable_memory_saver=False,
            )
            pool_tp8 = TpuFusedMHATokenToKVPool(
                size=1024, page_size=512, dtype=torch.float8_e4m3fn,
                head_num=2, head_dim=64, layer_num=1, device="cpu", enable_memory_saver=False,
            )
            pool_draft = TpuDraftFusedMHATokenToKVPool(
                size=1024, page_size=512, dtype=torch.float8_e4m3fn,
                head_num=1, head_dim=64, layer_num=1, device="cpu", enable_memory_saver=False,
            )
            self.assertTrue(pool_bd4.kv_pack_bd4)
            self.assertFalse(pool_bd4.kv_pack_dense2)
            self.assertTrue(_pool_kv_pack_bd4(pool_bd4))
            self.assertEqual(pool_bd4.kv_buffer[0].shape, (3, 128, 1, 4, 128))
            self.assertTrue(pool_bd4.kv_layout_tag().endswith("-bd4"))
            self.assertFalse(pool_tp8.kv_pack_bd4)
            self.assertFalse(pool_draft.kv_pack_bd4)
            self.assertIsNotNone(_get_write_kv_only_op(False, True))

            # Verify _logical_kv_view ordering against bd4 packing
            k_ref_u8 = np.random.default_rng(17).integers(1, 50, size=(3 * 512, 64), dtype=np.uint8)
            v_ref_u8 = np.random.default_rng(18).integers(51, 100, size=(3 * 512, 64), dtype=np.uint8)
            bd4_u8 = np.zeros((3, 128, 1, 4, 128), dtype=np.uint8)
            k_4 = k_ref_u8.reshape(3 * 128, 4, 64)
            v_4 = v_ref_u8.reshape(3 * 128, 4, 64)
            bd4_u8[:, :, 0, 0, :64] = k_4[:, 0, :].reshape(3, 128, 64)
            bd4_u8[:, :, 0, 0, 64:] = k_4[:, 1, :].reshape(3, 128, 64)
            bd4_u8[:, :, 0, 1, :64] = v_4[:, 0, :].reshape(3, 128, 64)
            bd4_u8[:, :, 0, 1, 64:] = v_4[:, 1, :].reshape(3, 128, 64)
            bd4_u8[:, :, 0, 2, :64] = k_4[:, 2, :].reshape(3, 128, 64)
            bd4_u8[:, :, 0, 2, 64:] = k_4[:, 3, :].reshape(3, 128, 64)
            bd4_u8[:, :, 0, 3, :64] = v_4[:, 2, :].reshape(3, 128, 64)
            bd4_u8[:, :, 0, 3, 64:] = v_4[:, 3, :].reshape(3, 128, 64)
            pool_bd4.kv_buffer[0].copy_(torch.from_numpy(bd4_u8).view(torch.float8_e4m3fn))
            k_got, v_got = pool_bd4.get_kv_buffer(0)
            self.assertTrue(np.array_equal(k_got.view(torch.uint8).numpy().reshape(-1, 64), k_ref_u8))
            self.assertTrue(np.array_equal(v_got.view(torch.uint8).numpy().reshape(-1, 64), v_ref_u8))

            seen = {}
            def _stub_rpa(*args, **kwargs):
                seen["kwargs"] = kwargs
                return args[0], args[3]
            RPA_V3_SPEC.invoke_kernel(
                _stub_rpa,
                queries=jax.ShapeDtypeStruct((4, 8, 64), jnp.bfloat16),
                keys=jax.ShapeDtypeStruct((4, 8, 64), jnp.bfloat16),
                values=jax.ShapeDtypeStruct((4, 8, 64), jnp.bfloat16),
                kv_cache_fused=jax.ShapeDtypeStruct((8, 128, 1, 4, 128), jnp.float8_e4m3fn),
                kv_lens=jax.ShapeDtypeStruct((4,), jnp.int32),
                page_indices=jax.ShapeDtypeStruct((8,), jnp.int32),
                cu_q_lens=jax.ShapeDtypeStruct((5,), jnp.int32),
                cu_kv_lens=jax.ShapeDtypeStruct((5,), jnp.int32),
                distribution=jax.ShapeDtypeStruct((3,), jnp.int32),
                attention_sink=None,
                sm_scale=64.0 ** -0.5,
                sliding_window=2049,
                soft_cap=None,
                k_scale=1.0,
                v_scale=1.0,
            )
            self.assertTrue(seen["kwargs"].get("_kv_pack_bd4"))
            self.assertNotIn("_kv_pack_dense2", seen["kwargs"])

        def _pack_bd4_u8_from_512(s512_u8: np.ndarray) -> np.ndarray:
            nb, ps_l = s512_u8.shape[0], s512_u8.shape[1]
            r = ps_l // 4
            x = s512_u8.reshape(nb, r, 4, 4, 128)
            out = np.empty((nb, r, 1, 4, 128), dtype=np.uint8)
            out[:, :, 0, 0, :64] = x[:, :, 0, 0, :64]
            out[:, :, 0, 0, 64:] = x[:, :, 1, 0, :64]
            out[:, :, 0, 1, :64] = x[:, :, 0, 1, :64]
            out[:, :, 0, 1, 64:] = x[:, :, 1, 1, :64]
            out[:, :, 0, 2, :64] = x[:, :, 2, 0, :64]
            out[:, :, 0, 2, 64:] = x[:, :, 3, 0, :64]
            out[:, :, 0, 3, :64] = x[:, :, 2, 1, :64]
            out[:, :, 0, 3, 64:] = x[:, :, 3, 1, :64]
            return out

        def _unpack_bd4_u8_to_kv64(bd4_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nb, r = bd4_u8.shape[0], bd4_u8.shape[1]
            k_out = np.empty((nb, r * 4, 64), dtype=np.uint8)
            v_out = np.empty((nb, r * 4, 64), dtype=np.uint8)
            k_out[:, 0::4, :] = bd4_u8[:, :, 0, 0, :64]
            k_out[:, 1::4, :] = bd4_u8[:, :, 0, 0, 64:]
            k_out[:, 2::4, :] = bd4_u8[:, :, 0, 2, :64]
            k_out[:, 3::4, :] = bd4_u8[:, :, 0, 2, 64:]
            v_out[:, 0::4, :] = bd4_u8[:, :, 0, 1, :64]
            v_out[:, 1::4, :] = bd4_u8[:, :, 0, 1, 64:]
            v_out[:, 2::4, :] = bd4_u8[:, :, 0, 3, :64]
            v_out[:, 3::4, :] = bd4_u8[:, :, 0, 3, 64:]
            return k_out, v_out

        rng = np.random.default_rng(1707)
        ps = 512

        # 2. Multi-step sequential RMW cache-write parity (kv_len mod 4 cycling 1->2->3->0) & write_kv_only
        bs_rmw, pps_rmw = 8, 2
        pre_lens = [509, 510, 511, 512, 1, 2, 3, 4]  # off_lo mod 4 in {1,2,3,0} & page cross
        s512_u8 = np.full((bs_rmw * pps_rmw + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        pi_rmw = np.arange(bs_rmw * pps_rmw, dtype=np.int32).reshape(bs_rmw, pps_rmw)
        rng.shuffle(pi_rmw.ravel())
        for s_i, pl_i in enumerate(pre_lens):
            for p_o in range((pl_i + ps - 1) // ps):
                pg = int(pi_rmw[s_i, p_o])
                rh = min(ps, pl_i - p_o * ps)
                s512_u8[pg, :rh, 0, :, :] = 0
                s512_u8[pg, :rh, 0, 0, :64] = _rand_fp8_u8(rng, (rh, 64))
                s512_u8[pg, :rh, 0, 1, :64] = _rand_fp8_u8(rng, (rh, 64))
        kvc_512 = _u8_to_fp8(s512_u8)
        kvc_bd4 = _u8_to_fp8(_pack_bd4_u8_from_512(s512_u8))
        pi_j = jnp.asarray(pi_rmw.ravel(), jnp.int32)
        cukv_j = jnp.arange(bs_rmw + 1, dtype=np.int32) * (pps_rmw * ps)
        cuq_j = jnp.arange(bs_rmw + 1, dtype=np.int32)
        dist_j = jnp.asarray([bs_rmw, bs_rmw, bs_rmw], jnp.int32)

        cur_lens = list(pre_lens)
        for step_i in range(4):
            cur_lens = [x + 1 for x in cur_lens]
            kvl_j = jnp.asarray(cur_lens, dtype=jnp.int32)
            q_s = jnp.asarray((rng.standard_normal((bs_rmw, 8, 64)) * 0.4).astype(np.float32), jnp.bfloat16)
            k_s = _u8_to_fp8(_rand_fp8_u8(rng, (bs_rmw, 1, 64)))
            v_s = _u8_to_fp8(_rand_fp8_u8(rng, (bs_rmw, 1, 64)))
            o_512, kvc_512 = run_decode_fast_rpa(
                q_s, k_s, v_s, kvc_512, kvl_j, pi_j, cuq_j, cukv_j, dist_j,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="tp16,w7,c1024,g4k,foldcol", decode_query_size=1,
                kv_pack_bd4=False, interpret=True,
            )
            o_bd4, kvc_bd4 = run_decode_fast_rpa(
                q_s, k_s, v_s, kvc_bd4, kvl_j, pi_j, cuq_j, cukv_j, dist_j,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="tp16,w7,c1024,g4k,foldcol", decode_query_size=1,
                kv_pack_bd4=True, interpret=True,
            )
            a5 = np.asarray(o_512, np.float32)
            ab = np.asarray(o_bd4, np.float32)
            self.assertTrue(np.all(np.isfinite(ab)))
            self.assertGreater(_cosine(ab, a5), 0.99999, f"step={step_i}")
            self.assertLessEqual(float(np.max(np.abs(ab - a5))), 1e-2, f"step={step_i}")
            u512 = np.asarray(kvc_512).view(np.uint8)
            k_b, v_b = _unpack_bd4_u8_to_kv64(np.asarray(kvc_bd4).view(np.uint8))
            for s_i, cl in enumerate(cur_lens):
                for tok in range(cl):
                    pg = int(pi_rmw[s_i, tok // ps])
                    off = tok % ps
                    self.assertTrue(np.array_equal(k_b[pg, off], u512[pg, off, 0, 0, :64]))
                    self.assertTrue(np.array_equal(v_b[pg, off], u512[pg, off, 0, 1, :64]))

        # write_kv_only_rpa_v3 across off_lo % 4 in {1, 2, 3, 0} + multi-page boundary
        q_lens_wk = [3, 5, 7, 2]
        pre_wk = [510, 509, 511, 512]
        tot_wk = [p + q for p, q in zip(pre_wk, q_lens_wk)]
        s_wk_512 = np.full((12, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        pi_wk = np.arange(8, dtype=np.int32).reshape(4, 2)
        rng.shuffle(pi_wk.ravel())
        for s_i, pl_i in enumerate(pre_wk):
            pg = int(pi_wk[s_i, 0])
            s_wk_512[pg, :pl_i, 0, :, :] = 0
            s_wk_512[pg, :pl_i, 0, 0, :64] = _rand_fp8_u8(rng, (pl_i, 64))
            s_wk_512[pg, :pl_i, 0, 1, :64] = _rand_fp8_u8(rng, (pl_i, 64))
        t_tot = sum(q_lens_wk)
        kv_new_u8 = np.zeros((t_tot, 1, 4, 128), dtype=np.uint8)
        kv_new_u8[:, 0, 0, :64] = _rand_fp8_u8(rng, (t_tot, 64))
        kv_new_u8[:, 0, 1, :64] = _rand_fp8_u8(rng, (t_tot, 64))
        cuq_wk = jnp.asarray([0] + list(np.cumsum(q_lens_wk)), jnp.int32)
        cukv_wk = jnp.arange(5, dtype=jnp.int32) * (2 * ps)
        kvl_wk = jnp.asarray(tot_wk, jnp.int32)
        pi_wk_j = jnp.asarray(pi_wk.ravel(), jnp.int32)
        dist_wk = jnp.asarray([4, 4, 4], jnp.int32)
        wk_512 = write_kv_only_rpa_v3(
            _u8_to_fp8(kv_new_u8), _u8_to_fp8(s_wk_512), kvl_wk, pi_wk_j, cuq_wk, cukv_wk, dist_wk,
            interpret=True, kv_pack_bd4=False,
        )
        wk_bd4 = write_kv_only_rpa_v3(
            _u8_to_fp8(kv_new_u8), _u8_to_fp8(_pack_bd4_u8_from_512(s_wk_512)), kvl_wk, pi_wk_j, cuq_wk, cukv_wk, dist_wk,
            interpret=True, kv_pack_bd4=True,
        )
        u_wk512 = np.asarray(wk_512).view(np.uint8)
        k_wkb, v_wkb = _unpack_bd4_u8_to_kv64(np.asarray(wk_bd4).view(np.uint8))
        for s_i, tl in enumerate(tot_wk):
            for tok in range(tl):
                pg = int(pi_wk[s_i, tok // ps])
                off = tok % ps
                self.assertTrue(np.array_equal(k_wkb[pg, off], u_wk512[pg, off, 0, 0, :64]))
                self.assertTrue(np.array_equal(v_wkb[pg, off], u_wk512[pg, off, 0, 1, :64]))

        # 3. RPAd_fast bd4 across batch4 SWA, batch2 SWA, Mix Full w_pg=17/33 (pps=17 non-mult-of-8), LC shared16, & Step (c)
        for case_tag, bs_c, sw_c, pps_c, mode_c, lens_c, n_act_c, shared_c in (
            (
                "swa_1k_batch4", 16, 2049, 2, "w8,batchfull,g4k,foldcol,batch4",
                [1, 3, 255, 511, 512, 513, 600, 750, 999, 1001, 1020, 1024, 333, 1, 1, 1], 13, None,
            ),
            (
                "swa_8k1k_batch2", 8, 2049, 5, "w8,s3,noadapt,g4k,foldcol,batch2",
                [515, 1111, 2049, 2300, 2559, 15, 7, 1], 7, None,
            ),
            (
                "full_w7_pps17", 4, None, 17, "tp16,w7,g4k,foldcol",
                [1201, 4097, 5123, 8600], 4, None,
            ),
            (
                "full_dyn_pps17", 4, None, 17, "tp16,v2,streamdyn,c512,foldcol",
                [513, 1205, 5121, 8600], 4, None,
            ),
            (
                "lc_shared16", 16, None, 4, "s4,c1024,shared16,q256,shsplit,pbf16,batchfull,foldcol,shfp8",
                [1500, 1503, 1511, 1601, 1700, 1800, 1900, 2001, 1400, 1300, 1200, 1100, 1025, 1026, 1, 1],
                14, 2,
            ),
        ):
            with self.subTest(regime=case_tag):
                s_c_u8 = np.full((bs_c * pps_c + 8, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
                pi_c = np.arange(bs_c * pps_c, dtype=np.int32).reshape(bs_c, pps_c)
                if shared_c is not None:
                    sh_pgs = np.arange(shared_c, dtype=np.int32) + (bs_c * pps_c)
                    pi_c[:, :shared_c] = sh_pgs[None, :]
                    s_c_u8[sh_pgs, :, 0, :, :] = 0
                    s_c_u8[sh_pgs, :, 0, 0, :64] = _rand_fp8_u8(rng, (shared_c, ps, 64))
                    s_c_u8[sh_pgs, :, 0, 1, :64] = _rand_fp8_u8(rng, (shared_c, ps, 64))
                for s_i in range(n_act_c):
                    old_l = lens_c[s_i] - 1
                    p0 = shared_c if shared_c is not None else 0
                    for p_o in range(p0, (old_l + ps - 1) // ps):
                        pg = int(pi_c[s_i, p_o])
                        rh = min(ps, old_l - p_o * ps)
                        s_c_u8[pg, :rh, 0, :, :] = 0
                        s_c_u8[pg, :rh, 0, 0, :64] = _rand_fp8_u8(rng, (rh, 64))
                        s_c_u8[pg, :rh, 0, 1, :64] = _rand_fp8_u8(rng, (rh, 64))
                kvc_5 = _u8_to_fp8(s_c_u8)
                kvc_4 = _u8_to_fp8(_pack_bd4_u8_from_512(s_c_u8))
                q8 = jnp.asarray((rng.standard_normal((bs_c, 8, 64)) * 0.4).astype(np.float32), jnp.bfloat16)
                k8 = _u8_to_fp8(_rand_fp8_u8(rng, (bs_c, 1, 64)))
                v8 = _u8_to_fp8(_rand_fp8_u8(rng, (bs_c, 1, 64)))
                pi_cj = jnp.asarray(pi_c.ravel(), jnp.int32)
                cuq_cj = jnp.arange(bs_c + 1, dtype=jnp.int32)
                cukv_cj = jnp.arange(bs_c + 1, dtype=jnp.int32) * (pps_c * ps)
                dist_cj = (
                    jnp.asarray([n_act_c, n_act_c, n_act_c, shared_c], jnp.int32)
                    if shared_c is not None
                    else jnp.asarray([n_act_c, n_act_c, n_act_c], jnp.int32)
                )
                kvl_cj = jnp.asarray(lens_c, jnp.int32)
                sh_arg = jnp.asarray([shared_c], jnp.int32) if shared_c is not None else None
                gate_c = jnp.asarray((rng.standard_normal((bs_c, 512)) * 0.4).astype(np.float32), jnp.bfloat16)
                q_prep = jnp.pad(q8, ((0, 0), (0, 0), (0, 64))).reshape(1, bs_c, 2, 4, 128)
                kv_prep = jnp.pad(jnp.concatenate([k8, v8], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs_c, 1, 4, 128)

                ref_m = mode_c.replace(",batch4", "").replace(",batch2", "")
                if shared_c is not None:
                    q16_r = jnp.pad(q8, ((0, 0), (0, 8), (0, 64))).reshape(1, bs_c, 8, 2, 128)
                    o_5, _ = run_decode_fast_rpa(
                        q16_r, kv_prep, None, kvc_5, kvl_cj, pi_cj, cuq_cj, cukv_cj, dist_cj,
                        sm_scale=64.0 ** -0.5, sliding_window=sw_c, k_scale=1.0, v_scale=1.0,
                        mode_str=mode_c, decode_query_size=1,
                        fused_out_pack=2, prepared_inputs=True, gate=gate_c, out_norm_eps=1e-5,
                        shared_pages=sh_arg, force_native_tp16=True, interpret=True,
                    )
                else:
                    o_5, _ = run_decode_fast_rpa(
                        q8, k8, v8, kvc_5, kvl_cj, pi_cj, cuq_cj, cukv_cj, dist_cj,
                        sm_scale=64.0 ** -0.5, sliding_window=sw_c, k_scale=1.0, v_scale=1.0,
                        mode_str=ref_m if "tp16" in ref_m else f"tp16,{ref_m}",
                        decode_query_size=1, fused_out_pack=2, prepared_inputs=False,
                        gate=gate_c, out_norm_eps=1e-5, kv_pack_bd4=False, interpret=True,
                    )
                o_4, _ = run_decode_fast_rpa(
                    q_prep, kv_prep, None, kvc_4, kvl_cj, pi_cj, cuq_cj, cukv_cj, dist_cj,
                    sm_scale=64.0 ** -0.5, sliding_window=sw_c, k_scale=1.0, v_scale=1.0,
                    mode_str=mode_c, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                    gate=gate_c, out_norm_eps=1e-5, shared_pages=sh_arg, kv_pack_bd4=True, interpret=True,
                )
                a5 = np.asarray(o_5[:n_act_c], np.float32)
                a4 = np.asarray(o_4[:n_act_c], np.float32)
                self.assertTrue(np.all(np.isfinite(np.asarray(o_4, np.float32))))
                self.assertGreater(_cosine(a4, a5), 0.99995, case_tag)
                self.assertLessEqual(float(np.max(np.abs(a4 - a5))), 2e-2, case_tag)

        # 4. RPAm prefill round-trip with bd4 (512 q tokens after 259 cached tokens -> off_lo%4=3 + multi-page RMW on 0x7F cache)
        interp_pm = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")
        t_pm, q_pm, cache_pm_l = 512, 512, 259
        kv_pm_l = cache_pm_l + q_pm
        s_pm_512 = np.full((4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        s_pm_512[0, :cache_pm_l, 0, :, :] = 0
        s_pm_512[0, :cache_pm_l, 0, 0, :64] = _rand_fp8_u8(rng, (cache_pm_l, 64))
        s_pm_512[0, :cache_pm_l, 0, 1, :64] = _rand_fp8_u8(rng, (cache_pm_l, 64))
        q_pm_np = (rng.standard_normal((1, t_pm, 4, 2, 128)) * 0.4).astype(np.float32)
        q_pm_np[..., 64:] = 0.0
        q_pm_arr = jnp.asarray(q_pm_np, jnp.bfloat16)
        kv_pm_u8 = np.zeros((t_pm, 1, 4, 128), dtype=np.uint8)
        kv_pm_u8[:, 0, 0, :64] = _rand_fp8_u8(rng, (t_pm, 64))
        kv_pm_u8[:, 0, 1, :64] = _rand_fp8_u8(rng, (t_pm, 64))
        kv_pm_arr = _u8_to_fp8(kv_pm_u8)
        pm_opts = frozenset(
            {"stationary_q_m0_ring_fuse", "skip_empty_cdma", "mixed_s3", "tp16_sq", "mk128", "la2", "sub8_l", "bf16_pop", "unroll_up", "fp8_qk"}
        )
        o_pm5, c_pm5 = ragged_paged_attention(
            q_pm_arr, kv_pm_arr, None, _u8_to_fp8(s_pm_512),
            jnp.asarray([kv_pm_l], jnp.int32), jnp.asarray([0, 1], jnp.int32),
            jnp.asarray([0, q_pm], jnp.int32), jnp.asarray([0, 2 * ps], jnp.int32),
            jnp.asarray([0, 1, 1], jnp.int32), None,
            sm_scale=64.0 ** -0.5, sliding_window=2049, chunk_prefill_size=None,
            m_block_sizes=(512, 2048, 256, 512), _prefill_opts=pm_opts,
            _fused_out_pack=-2, _prepared_inputs=True, _skip_decode=True, _a3_out_layout=True,
            _kv_pack_bd4=False, _interpret=interp_pm,
        )
        o_pm4, c_pm4 = ragged_paged_attention(
            q_pm_arr, kv_pm_arr, None, _u8_to_fp8(_pack_bd4_u8_from_512(s_pm_512)),
            jnp.asarray([kv_pm_l], jnp.int32), jnp.asarray([0, 1], jnp.int32),
            jnp.asarray([0, q_pm], jnp.int32), jnp.asarray([0, 2 * ps], jnp.int32),
            jnp.asarray([0, 1, 1], jnp.int32), None,
            sm_scale=64.0 ** -0.5, sliding_window=2049, chunk_prefill_size=None,
            m_block_sizes=(512, 2048, 256, 512), _prefill_opts=pm_opts,
            _fused_out_pack=-2, _prepared_inputs=True, _skip_decode=True, _a3_out_layout=True,
            _kv_pack_bd4=True, _interpret=interp_pm,
        )
        self.assertTrue(np.all(np.isfinite(np.asarray(o_pm4, np.float32))))
        self.assertGreater(_cosine(np.asarray(o_pm4, np.float32), np.asarray(o_pm5, np.float32)), 0.99999)
        self.assertLessEqual(float(np.max(np.abs(np.asarray(o_pm4, np.float32) - np.asarray(o_pm5, np.float32)))), 1e-2)
        u_pm5 = np.asarray(c_pm5).view(np.uint8)
        k_pm4, v_pm4 = _unpack_bd4_u8_to_kv64(np.asarray(c_pm4).view(np.uint8))
        for tok in range(kv_pm_l):
            pg, off = tok // ps, tok % ps
            self.assertTrue(np.array_equal(k_pm4[pg, off], u_pm5[pg, off, 0, 0, :64]))
            self.assertTrue(np.array_equal(v_pm4[pg, off], u_pm5[pg, off, 0, 1, :64]))

        # 5. Prestarted Step (c) parity with bd4
        bs_s, pps_s, n_act_s = 16, 2, 13
        mode_s = "w8,s3,batchfull,g4k,foldcol,batch4"
        cfg_s = parse_decode_fast_config(
            mode_s, sliding_window=2049, pages_per_seq=pps_s, page_size=ps,
            decode_query_size=1, kv_pack_bd4=True,
        )
        lens_s = [120, 480, 513, 900, 1024] * 2 + [1, 300, 600] + [1, 1, 1]
        s5_u8 = np.full((bs_s * pps_s + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        pi_s = np.arange(bs_s * pps_s, dtype=np.int32).reshape(bs_s, pps_s)
        for s_i in range(n_act_s):
            old_l = lens_s[s_i] - 1
            for p_o in range((old_l + ps - 1) // ps):
                pg = int(pi_s[s_i, p_o])
                rh = min(ps, old_l - p_o * ps)
                s5_u8[pg, :rh, 0, :, :] = 0
                s5_u8[pg, :rh, 0, 0, :64] = _rand_fp8_u8(rng, (rh, 64))
                s5_u8[pg, :rh, 0, 1, :64] = _rand_fp8_u8(rng, (rh, 64))
        kvc_s4 = _u8_to_fp8(_pack_bd4_u8_from_512(s5_u8))
        q8_s = jnp.asarray((rng.standard_normal((bs_s, 8, 64)) * 0.4).astype(np.float32), jnp.bfloat16)
        q_bd_s = jnp.concatenate([jnp.pad(q8_s, ((0, 0), (0, 0), (0, 64))), jnp.pad(q8_s, ((0, 0), (0, 0), (64, 0)))], axis=1)
        k8_s = _u8_to_fp8(_rand_fp8_u8(rng, (bs_s, 1, 64)))
        v8_s = _u8_to_fp8(_rand_fp8_u8(rng, (bs_s, 1, 64)))
        kv_p_s = jnp.pad(jnp.concatenate([k8_s, v8_s], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs_s, 1, 4, 128)
        gate_s = jnp.asarray((rng.standard_normal((bs_s, 512)) * 0.4).astype(np.float32), jnp.bfloat16)
        pi_sj = jnp.asarray(pi_s.ravel(), jnp.int32)
        cuq_sj = jnp.arange(bs_s + 1, dtype=jnp.int32)
        cukv_sj = jnp.arange(bs_s + 1, dtype=np.int32) * (pps_s * ps)
        dist_sj = jnp.asarray([n_act_s, n_act_s, n_act_s], jnp.int32)
        kvl_sj = jnp.asarray(lens_s, jnp.int32)
        o_std_s, kv_std_s = run_decode_fast_rpa(
            q8_s, k8_s, v8_s, kvc_s4, kvl_sj, pi_sj, cuq_sj, cukv_sj, dist_sj,
            sm_scale=64.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
            mode_str=mode_s, decode_query_size=1, fused_out_pack=2,
            gate=gate_s, out_norm_eps=1e-5, kv_pack_bd4=True, interpret=True,
        )

        def _pre_bd4(klr, pir, cqr, ckvr, dr, q_r, kv_r, g_r, chbm, oout, upd, bkv_r, st_r, o_r, kw_r, init_r, sm_r, s1, s2):
            pre_st = prepare_and_start_rpad_dmas(
                kv_lens_ref=klr, page_indices_ref=pir, cu_q_lens_ref=cqr, cu_kv_lens_ref=ckvr,
                distribution_ref=dr, kv_cache_hbm_ref=chbm, bkv_vmem_ref=bkv_r, bkv_sems=s1,
                sliding_window=2049, cfg=cfg_s, one_block_per_seq=True, decode_query_size=1,
                kv_pack_bd4=True, interpret_mode=True,
            )
            _run_rpad_vmem_core(
                kv_lens_ref=klr, page_indices_ref=pir, cu_q_lens_ref=cqr, cu_kv_lens_ref=ckvr,
                distribution_ref=dr, q_vmem_ref=q_r, kv_merged_vmem_ref=kv_r, gate_vmem_ref=g_r,
                kv_cache_hbm_ref=chbm, o_out_vmem_ref=oout, updated_kv_cache_hbm_ref=upd,
                bkv_vmem_ref=bkv_r, state_vmem_ref=st_r, o_vmem_ref=o_r, kv_write_vmem_ref=kw_r,
                init_sv_vmem_ref=init_r, written_cnt_smem_ref=sm_r, bkv_sems=s1, cache_write_sem=s2,
                sm_scale=64.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0, cfg=cfg_s,
                one_block_per_seq=True, decode_query_size=1, in_kernel_pack2=True, has_gate=True,
                out_norm_eps=1e-5, kv_pack_bd4=True, interpret_mode=True, prestarted_bkv_state=pre_st,
            )

        o_pre_s, kv_pre_s = pl.pallas_call(
            _pre_bd4,
            out_shape=[jax.ShapeDtypeStruct((bs_s, 512), jnp.bfloat16), jax.ShapeDtypeStruct(kvc_s4.shape, kvc_s4.dtype)],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=5,
                in_specs=[
                    pl.BlockSpec(q_bd_s.shape, lambda *_: (0, 0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(kv_p_s.shape, lambda *_: (0, 0, 0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(gate_s.shape, lambda *_: (0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=[
                    pl.BlockSpec((bs_s, 512), lambda *_: (0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM((cfg_s.num_slots, 4 * cfg_s.max_ring_pages, 128, 1, 4, 128), kvc_s4.dtype),
                    pltpu.VMEM((3, 8, 128), jnp.float32),
                    pltpu.VMEM((bs_s, 8, 128), jnp.bfloat16),
                    pltpu.VMEM((bs_s * 3, 8, 1, 4, 128), kvc_s4.dtype),
                    pltpu.VMEM((2, ((bs_s + 31) // 32) * 32, 8, 128), jnp.float32),
                    pltpu.SMEM((2 + cfg_s.num_slots,), jnp.int32),
                    pltpu.SemaphoreType.DMA((cfg_s.num_slots,)),
                    pltpu.SemaphoreType.DMA((1,)),
                ],
            ),
            input_output_aliases={8: 1},
            interpret=pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan"),
        )(kvl_sj, pi_sj, cuq_sj, cukv_sj, dist_sj, q_bd_s, kv_p_s, gate_s, kvc_s4)
        self.assertTrue(np.array_equal(np.asarray(o_pre_s).view(np.uint16), np.asarray(o_std_s).view(np.uint16)))
        self.assertTrue(np.array_equal(np.asarray(kv_pre_s).view(np.uint8), np.asarray(kv_std_s).view(np.uint8)))

        # 6. TPU v7 AOT export across all bd4 regimes + write_kv_only_rpa_v3
        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            for tag_e, bs_e, sw_e, pps_e, mode_e, sh_e in (
                ("bd4_swa_1k_b4", 64, 2049, 2, "w8,batchfull,g4k,foldcol,batch4", False),
                ("bd4_swa_8k1k_b2", 64, 2049, 5, "w8,s3,noadapt,g4k,foldcol,batch2", False),
                ("bd4_full_w7_pps17", 64, None, 17, "tp16,w7,g4k,foldcol", False),
                ("bd4_lc_sh64", 64, None, 256, "s4,c2048,shared64,q1024,shsplit,pbf16,batchfull,foldcol,shfp8", True),
            ):
                with self.subTest(export=tag_e):
                    d_len = 4 if sh_e else 3
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                                q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                                sm_scale=64.0 ** -0.5, sliding_window=sw_e, k_scale=1.0, v_scale=1.0,
                                mode_str=mode_e, decode_query_size=1, fused_out_pack=2,
                                prepared_inputs=True, gate=g_, out_norm_eps=1e-5, kv_pack_bd4=True,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, bs_e, 2, 4, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs_e, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs_e, 512), jnp.bfloat16),
                        jax.ShapeDtypeStruct((bs_e * pps_e + 8, 128, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((bs_e,), jnp.int32),
                        jax.ShapeDtypeStruct((bs_e * pps_e,), jnp.int32),
                        jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((bs_e + 1,), jnp.int32),
                        jax.ShapeDtypeStruct((d_len,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("tpu_custom_call", hlo.lower())
                    self.assertIn("-bd4", hlo)
            hlo_wk = jax_export.export(
                jax.jit(
                    lambda kvn_, c_, kl_, pi_, cq_, ckv_, d_: write_kv_only_rpa_v3(
                        kvn_, c_, kl_, pi_, cq_, ckv_, d_, kv_pack_bd4=True,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((64, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((128, 128, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((16,), jnp.int32),
                jax.ShapeDtypeStruct((64,), jnp.int32),
                jax.ShapeDtypeStruct((17,), jnp.int32),
                jax.ShapeDtypeStruct((17,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("RPA_write_kv_only_bd4-p_512", hlo_wk)

            # 7d (AOT). TPU v7 export for fp8dp / streamdyn c128 (pad_rows) / clean1p ps=128
            for tag_e2, ps_e2, sw_e2, pps_e2, mode_e2 in (
                ("bd4_w7_fp8dp", 512, None, 5, "tp16,w7,g4k,foldcol,fp8dp"),
                ("bd4_streamdyn_c128", 128, None, 4, "tp16,v2,streamdyn,c128,foldcol,fp8dp"),
                ("bd4_clean1p_ps128", 128, 2049, 17, "tp16,clean1p"),
            ):
                with self.subTest(export=tag_e2):
                    hlo2 = jax_export.export(
                        jax.jit(
                            lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_, m_=mode_e2, sw_=sw_e2: run_decode_fast_rpa(
                                q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                                sm_scale=64.0 ** -0.5, sliding_window=sw_, k_scale=1.0, v_scale=1.0,
                                mode_str=m_, decode_query_size=1, fused_out_pack=2,
                                prepared_inputs=True, gate=g_, out_norm_eps=1e-5, kv_pack_bd4=True,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, 8, 2, 4, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((8, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((8, 512), jnp.bfloat16),
                        jax.ShapeDtypeStruct((8 * pps_e2 + 4, ps_e2 // 4, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((8,), jnp.int32),
                        jax.ShapeDtypeStruct((8 * pps_e2,), jnp.int32),
                        jax.ShapeDtypeStruct((9,), jnp.int32),
                        jax.ShapeDtypeStruct((9,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("tpu_custom_call", hlo2.lower())
                    self.assertIn("-bd4", hlo2)

        # 7a. kv_merged with 0x7F NaNs in padded lanes 64..127 (hoistkv & nohoistkv & dq=2)
        for tag_nan, mode_nan, dq_nan in (
            ("hoist_b4", "w8,batchfull,g4k,foldcol,batch4", 1),
            ("nohoist_w7", "tp16,w7,nohoistkv,c1024,g4k,foldcol", 1),
            ("multiq_dq2", "tp16,v2,multiq,stream1p,verify", 2),
        ):
            with self.subTest(nan_pad=tag_nan):
                bs_n, pps_n = 4, 2
                n_tok_n = bs_n * dq_nan
                s_n_u8 = np.zeros((bs_n * pps_n + 4, ps, 1, 4, 128), dtype=np.uint8)
                lens_n = [10 * dq_nan + i * 31 for i in range(bs_n)]
                pi_n = np.arange(bs_n * pps_n, dtype=np.int32).reshape(bs_n, pps_n)
                for s_i, cl in enumerate(lens_n):
                    rh = cl - dq_nan
                    s_n_u8[pi_n[s_i, 0], :rh, 0, 0, :64] = _rand_fp8_u8(rng, (rh, 64))
                    s_n_u8[pi_n[s_i, 0], :rh, 0, 1, :64] = _rand_fp8_u8(rng, (rh, 64))
                kvc_n = _u8_to_fp8(_pack_bd4_u8_from_512(s_n_u8))
                q_n = jnp.asarray((rng.standard_normal((n_tok_n, 8, 64)) * 0.4).astype(np.float32), jnp.bfloat16)
                q_n_prep = jnp.pad(q_n, ((0, 0), (0, 0), (0, 64))).reshape(1, n_tok_n, 2, 4, 128)
                kv_clean_u8 = np.zeros((n_tok_n, 1, 4, 128), dtype=np.uint8)
                kv_clean_u8[:, 0, 0, :64] = _rand_fp8_u8(rng, (n_tok_n, 64))
                kv_clean_u8[:, 0, 1, :64] = _rand_fp8_u8(rng, (n_tok_n, 64))
                kv_nan_u8 = kv_clean_u8.copy()
                kv_nan_u8[:, 0, :2, 64:] = 0x7F
                pi_nj = jnp.asarray(pi_n.ravel(), jnp.int32)
                cuq_nj = jnp.arange(bs_n + 1, dtype=jnp.int32) * dq_nan
                cukv_nj = jnp.arange(bs_n + 1, dtype=jnp.int32) * (pps_n * ps)
                dist_nj = jnp.asarray([bs_n, bs_n, bs_n], jnp.int32)
                kvl_nj = jnp.asarray(lens_n, jnp.int32)
                o_clean, _ = run_decode_fast_rpa(
                    q_n_prep, _u8_to_fp8(kv_clean_u8), None, kvc_n, kvl_nj, pi_nj, cuq_nj, cukv_nj, dist_nj,
                    sm_scale=64.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                    mode_str=mode_nan, decode_query_size=dq_nan, prepared_inputs=True,
                    kv_pack_bd4=True, interpret=True,
                )
                o_nan, _ = run_decode_fast_rpa(
                    q_n_prep, _u8_to_fp8(kv_nan_u8), None, kvc_n, kvl_nj, pi_nj, cuq_nj, cukv_nj, dist_nj,
                    sm_scale=64.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                    mode_str=mode_nan, decode_query_size=dq_nan, prepared_inputs=True,
                    kv_pack_bd4=True, interpret=True,
                )
                self.assertTrue(np.all(np.isfinite(np.asarray(o_nan, np.float32))))
                self.assertTrue(np.array_equal(np.asarray(o_nan).view(np.uint16), np.asarray(o_clean).view(np.uint16)))

        # 7b. u4flat guard + fp8dp (w7auto & streamdyn c128) + clean1p (ps=128)
        for tag_f, ps_f, sw_f, pps_f, mode_f, lens_f in (
            ("u4flat_guard", 512, 2049, 2, "tp16,w8,s4,u4flat", [100, 513, 700, 1024]),
            ("w7_fp8dp", 512, None, 5, "tp16,w7,g4k,foldcol,fp8dp", [200, 700, 1200, 2100]),
            ("streamdyn_c128_fp8dp", 128, None, 4, "tp16,v2,streamdyn,c128,foldcol,fp8dp", [65, 193, 321, 500]),
            ("clean1p_ps128", 128, 2049, 17, "tp16,clean1p", [129, 1025, 2049, 2176]),
        ):
            with self.subTest(fix_regime=tag_f):
                bs_f = len(lens_f)
                s_f_u8 = np.full((bs_f * pps_f + 4, ps_f, 1, 4, 128), 0x7F, dtype=np.uint8)
                pi_f = np.arange(bs_f * pps_f, dtype=np.int32).reshape(bs_f, pps_f)
                for s_i, cl in enumerate(lens_f):
                    old_l = cl - 1
                    for p_o in range((old_l + ps_f - 1) // ps_f):
                        pg = int(pi_f[s_i, p_o])
                        rh = min(ps_f, old_l - p_o * ps_f)
                        s_f_u8[pg, :rh, 0, :, :] = 0
                        s_f_u8[pg, :rh, 0, 0, :64] = _rand_fp8_u8(rng, (rh, 64))
                        s_f_u8[pg, :rh, 0, 1, :64] = _rand_fp8_u8(rng, (rh, 64))
                kvc_f5 = _u8_to_fp8(s_f_u8)
                kvc_f4 = _u8_to_fp8(_pack_bd4_u8_from_512(s_f_u8))
                q_f = jnp.asarray((rng.standard_normal((bs_f, 8, 64)) * 0.4).astype(np.float32), jnp.bfloat16)
                k_f = _u8_to_fp8(_rand_fp8_u8(rng, (bs_f, 1, 64)))
                v_f = _u8_to_fp8(_rand_fp8_u8(rng, (bs_f, 1, 64)))
                pi_fj = jnp.asarray(pi_f.ravel(), jnp.int32)
                cuq_fj = jnp.arange(bs_f + 1, dtype=jnp.int32)
                cukv_fj = jnp.arange(bs_f + 1, dtype=jnp.int32) * (pps_f * ps_f)
                dist_fj = jnp.asarray([bs_f, bs_f, bs_f], jnp.int32)
                kvl_fj = jnp.asarray(lens_f, jnp.int32)
                o_f5, c_f5 = run_decode_fast_rpa(
                    q_f, k_f, v_f, kvc_f5, kvl_fj, pi_fj, cuq_fj, cukv_fj, dist_fj,
                    sm_scale=64.0 ** -0.5, sliding_window=sw_f, k_scale=1.0, v_scale=1.0,
                    mode_str=mode_f, decode_query_size=1, kv_pack_bd4=False, interpret=True,
                )
                o_f4, c_f4 = run_decode_fast_rpa(
                    q_f, k_f, v_f, kvc_f4, kvl_fj, pi_fj, cuq_fj, cukv_fj, dist_fj,
                    sm_scale=64.0 ** -0.5, sliding_window=sw_f, k_scale=1.0, v_scale=1.0,
                    mode_str=mode_f, decode_query_size=1, kv_pack_bd4=True, interpret=True,
                )
                self.assertTrue(np.all(np.isfinite(np.asarray(o_f4, np.float32))))
                self.assertGreater(_cosine(np.asarray(o_f4, np.float32), np.asarray(o_f5, np.float32)), 0.999 if "fp8dp" in tag_f else 0.99995, tag_f)
                self.assertLessEqual(float(np.max(np.abs(np.asarray(o_f4, np.float32) - np.asarray(o_f5, np.float32)))), 0.1 if "fp8dp" in tag_f else 2e-2, tag_f)
                u_f5 = np.asarray(c_f5).view(np.uint8)
                k_f4, v_f4 = _unpack_bd4_u8_to_kv64(np.asarray(c_f4).view(np.uint8))
                for s_i, cl in enumerate(lens_f):
                    for tok in range(cl):
                        pg, off = int(pi_f[s_i, tok // ps_f]), tok % ps_f
                        self.assertTrue(np.array_equal(k_f4[pg, off], u_f5[pg, off, 0, 0, :64]), tag_f)
                        self.assertTrue(np.array_equal(v_f4[pg, off], u_f5[pg, off, 0, 1, :64]), tag_f)

        # 7c. shared16 use_flat_priv=False mixed _only_shared (pos%4 != 0) + _private group
        bs_sh, n_act_sh, pps_sh, sh_pgs_n = 16, 4, 3, 1
        mode_sh = "s4,c1024,shared16,q256,shsplit,pbf16,batchfull,foldcol"
        cfg_sh = parse_decode_fast_config(
            mode_sh, sliding_window=None, pages_per_seq=pps_sh, page_size=ps,
            decode_query_size=1, shared_group=16, kv_pack_bd4=True,
        )
        lens_sh = [510, 1500, 700, 900] + [1] * (bs_sh - n_act_sh)  # Seq 0: total_pgs=1<=shared=1 with (kv_len-1)%4==1; Seq 1..3: private
        s_sh_u8 = np.full((bs_sh * pps_sh + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        pi_sh = np.arange(bs_sh * pps_sh, dtype=np.int32).reshape(bs_sh, pps_sh)
        sh_pg_id = bs_sh * pps_sh
        pi_sh[:, 0] = sh_pg_id
        s_sh_u8[sh_pg_id, :, 0, :, :] = 0
        s_sh_u8[sh_pg_id, :, 0, 0, :64] = _rand_fp8_u8(rng, (ps, 64))
        s_sh_u8[sh_pg_id, :, 0, 1, :64] = _rand_fp8_u8(rng, (ps, 64))
        for s_i in range(n_act_sh):
            old_l = lens_sh[s_i] - 1
            for p_o in range(1, (old_l + ps - 1) // ps):
                pg = int(pi_sh[s_i, p_o])
                rh = min(ps, old_l - p_o * ps)
                s_sh_u8[pg, :rh, 0, :, :] = 0
                s_sh_u8[pg, :rh, 0, 0, :64] = _rand_fp8_u8(rng, (rh, 64))
                s_sh_u8[pg, :rh, 0, 1, :64] = _rand_fp8_u8(rng, (rh, 64))
        kvc_sh5 = _u8_to_fp8(s_sh_u8)
        kvc_sh4 = _u8_to_fp8(_pack_bd4_u8_from_512(s_sh_u8))
        q_sh = jnp.asarray((rng.standard_normal((bs_sh, 8, 64)) * 0.4).astype(np.float32), jnp.bfloat16)
        q_sh_bd = jnp.concatenate([jnp.pad(q_sh, ((0, 0), (0, 0), (0, 64))), jnp.pad(q_sh, ((0, 0), (0, 0), (64, 0)))], axis=1)
        q_sh_16 = jnp.pad(q_sh, ((0, 0), (0, 8), (0, 64))).reshape(1, bs_sh, 8, 2, 128)
        k_sh = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sh, 1, 64)))
        v_sh = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sh, 1, 64)))
        kv_sh_p = jnp.pad(jnp.concatenate([k_sh, v_sh], axis=1), ((0, 0), (0, 2), (0, 64))).reshape(bs_sh, 1, 4, 128)
        kv_sh_pad32 = jnp.pad(kv_sh_p, ((0, 32 - bs_sh), (0, 0), (0, 0), (0, 0)))
        pi_shj = jnp.asarray(pi_sh.ravel(), jnp.int32)
        cuq_shj = jnp.arange(bs_sh + 1, dtype=jnp.int32)
        cukv_shj = jnp.arange(bs_sh + 1, dtype=jnp.int32) * (pps_sh * ps)
        dist_shj = jnp.asarray([n_act_sh, n_act_sh, n_act_sh, sh_pgs_n], jnp.int32)
        kvl_shj = jnp.asarray(lens_sh, jnp.int32)
        o_sh5, _ = run_decode_fast_rpa(
            q_sh_16, kv_sh_p, None, kvc_sh5, kvl_shj, pi_shj, cuq_shj, cukv_shj, dist_shj,
            sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str=mode_sh, decode_query_size=1, prepared_inputs=True,
            shared_pages=jnp.asarray([sh_pgs_n], jnp.int32), force_native_tp16=True, interpret=True,
        )

        def _sh_no_flat(klr, pir, cqr, ckvr, dr, q_r, kv_r, chbm, oout, upd, bkv_r, st_r, o_r, kw_r, sm_r, s1, s2):
            _run_rpad_vmem_core(
                kv_lens_ref=klr, page_indices_ref=pir, cu_q_lens_ref=cqr, cu_kv_lens_ref=ckvr,
                distribution_ref=dr, q_vmem_ref=q_r, kv_merged_vmem_ref=kv_r, gate_vmem_ref=None,
                kv_cache_hbm_ref=chbm, o_out_vmem_ref=oout, updated_kv_cache_hbm_ref=upd,
                bkv_vmem_ref=bkv_r, state_vmem_ref=st_r, o_vmem_ref=o_r, kv_write_vmem_ref=kw_r,
                init_sv_vmem_ref=None, written_cnt_smem_ref=sm_r, bkv_sems=s1, cache_write_sem=s2,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0, cfg=cfg_sh,
                one_block_per_seq=False, decode_query_size=1, in_kernel_pack2=False, has_gate=False,
                out_norm_eps=None, kv_pack_bd4=True, interpret_mode=True,
            )

        o_sh4, c_sh4 = pl.pallas_call(
            _sh_no_flat,
            out_shape=[jax.ShapeDtypeStruct((bs_sh, 8, 128), jnp.bfloat16), jax.ShapeDtypeStruct(kvc_sh4.shape, kvc_sh4.dtype)],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=5,
                in_specs=[
                    pl.BlockSpec(q_sh_bd.shape, lambda *_: (0, 0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(kv_sh_pad32.shape, lambda *_: (0, 0, 0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=[
                    pl.BlockSpec((bs_sh, 8, 128), lambda *_: (0, 0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM((cfg_sh.num_slots, cfg_sh.max_ring_pages, ps // 4, 1, 4, 128), kvc_sh4.dtype),
                    pltpu.VMEM((3, cfg_sh.shared_group_size * 8, 128), jnp.float32),
                    pltpu.VMEM((bs_sh, 8, 128), jnp.bfloat16),
                    pltpu.VMEM((bs_sh * 3, 8, 1, 4, 128), kvc_sh4.dtype),
                    pltpu.SMEM((2 + cfg_sh.num_slots,), jnp.int32),  # priv_smem_cap==0 -> forces use_flat_priv=False
                    pltpu.SemaphoreType.DMA((cfg_sh.num_slots,)),
                    pltpu.SemaphoreType.DMA((1,)),
                ],
            ),
            input_output_aliases={7: 1},
            interpret=pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan"),
        )(kvl_shj, pi_shj, cuq_shj, cukv_shj, dist_shj, q_sh_bd, kv_sh_pad32, kvc_sh4)
        o_sh4_s = np.asarray(o_sh4[:n_act_sh, :, :64], np.float32)
        o_sh5_s = np.asarray(o_sh5, np.float32).reshape(bs_sh, 8, 64)[:n_act_sh]
        self.assertGreater(_cosine(o_sh4_s, o_sh5_s), 0.99995)
        k_sh4, v_sh4 = _unpack_bd4_u8_to_kv64(np.asarray(c_sh4).view(np.uint8))
        kv_np = np.asarray(kv_sh_p).view(np.uint8)
        for s_i in range(n_act_sh):
            cl = lens_sh[s_i]
            pg, off = int(pi_sh[s_i, (cl - 1) // ps]), (cl - 1) % ps
            self.assertTrue(np.array_equal(k_sh4[pg, off], kv_np[s_i, 0, 0, :64]))
            self.assertTrue(np.array_equal(v_sh4[pg, off], kv_np[s_i, 0, 1, :64]))
        self.assertTrue(np.array_equal(k_sh4[sh_pg_id, 508], s_sh_u8[sh_pg_id, 508, 0, 0, :64]))
        self.assertTrue(np.array_equal(v_sh4[sh_pg_id, 508], s_sh_u8[sh_pg_id, 508, 0, 1, :64]))

        # 7d. Cross-caller bd4 guards: dflash_draft_kv_write & maybe_forward_m2m_decode
        from unittest.mock import MagicMock
        from sglang_tpu.dflash_draft_kv_write import _scatter_layer_kv, dflash_write_draft_kv
        from sglang_tpu.muse_megakernel.decode.m2m.ops import maybe_forward_m2m_decode

        mock_pool = MagicMock(kv_pack_heads=1, kv_pack_dense2=False, kv_pack_bd4=True)
        with self.assertRaisesRegex(NotImplementedError, "non-bd4"):
            dflash_write_draft_kv(
                draft_model=MagicMock(), kv_pool=mock_pool, ctx_hidden=torch.zeros(1, 8),
                ctx_positions=torch.zeros(1, dtype=torch.long), ctx_cache_loc=torch.zeros(1, dtype=torch.long),
                cache_loc_2d=None, commit_lens=None, max_num_reqs=1,
            )
        with self.assertRaisesRegex(NotImplementedError, "non-bd4"):
            _scatter_layer_kv(
                kv_pool=mock_pool, layer_idx=0, loc=torch.zeros(1, dtype=torch.long),
                k=torch.zeros(1, 1, 64), v=torch.zeros(1, 1, 64),
            )
        mock_m2m_model = MagicMock()
        mock_m2m_model.pp_group.is_first_rank = True
        mock_m2m_model.pp_group.is_last_rank = True
        mock_m2m_model.layers_to_capture = None
        mock_m2m_model.__dict__["_decode_col_bridges"] = 3
        mock_m2m_model.start_layer = 0
        mock_m2m_model.end_layer = 1
        mock_m2m_backend = MagicMock(
            _dcp=1, kv_cache_quant_dtype=torch.float8_e4m3fn, _kv_pack_dense2=False, _kv_pack_bd4=True,
        )
        mock_m2m_backend.forward_metadata.decode_query_size = 1
        mock_m2m_backend.forward_metadata.cp_write_segments = None
        mock_m2m_backend._kv_lens.return_value = torch.ones(16, dtype=torch.int32)
        with (
            patch("sglang_tpu.decode_qkv_epilogue_tpu._resolve_decode_attn_backend", return_value=mock_m2m_backend),
            patch("sglang_tpu.models.muse_spark._layer_tp_size", return_value=16),
        ):
            self.assertIsNone(
                maybe_forward_m2m_decode(
                    mock_m2m_model,
                    positions=torch.arange(16, dtype=torch.int32),
                    stream=torch.zeros(16, 128),
                    carried_attn_input=None,
                    cos_sin_step=None,
                    forward_batch=MagicMock(),
                    tok_sharded_stream=False,
                )
            )
            mock_m2m_backend._kv_lens.assert_called_once()

    def test_cand18_tp8_d2grp_shared_hoistkv_c256_and_lc_step_c_swa(self):
        """Verify TP=8 dense2+batch4/2, shared16/64 hoistkv, c256 batch4, and LC SWA Step (c) plan."""
        import os
        from unittest.mock import patch

        import jax
        from jax import export as jax_export
        import jax.numpy as jnp
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu
        import numpy as np

        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            _run_rpad_vmem_core,
            can_use_swabatch_group_runtime,
            parse_decode_fast_config,
            prepare_and_start_rpad_dmas,
            run_decode_fast_rpa,
        )
        from sglang_tpu.envs import envs
        from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
            resolve_step_c_rpa_plan,
        )

        # 1. Default-OFF invariants
        self.assertFalse(envs.SGLANG_TPU_RPA_V3_DENSE2_GROUP.get())
        self.assertFalse(envs.SGLANG_TPU_RPA_V3_SHARED_HOISTKV.get())
        self.assertFalse(envs.SGLANG_TPU_MUSE_MK_STEP_C_LC_SWA.get())

        def _pack_dense2_u8(src_u8: np.ndarray) -> np.ndarray:
            nb, ps_l = src_u8.shape[0], src_u8.shape[1]
            r = ps_l // 2
            x = src_u8.reshape(nb, r, 2, 4, 128)
            out = np.empty((nb, r, 1, 4, 128), dtype=np.uint8)
            out[:, :, 0, 0, :] = x[:, :, 0, 0, :]
            out[:, :, 0, 1, :] = x[:, :, 0, 1, :]
            out[:, :, 0, 2, :] = x[:, :, 1, 0, :]
            out[:, :, 0, 3, :] = x[:, :, 1, 1, :]
            return out

        def _unpack_dense2_u8(d2_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nb, r = d2_u8.shape[0], d2_u8.shape[1]
            k_out = np.empty((nb, r * 2, 128), dtype=np.uint8)
            v_out = np.empty((nb, r * 2, 128), dtype=np.uint8)
            k_out[:, 0::2, :] = d2_u8[:, :, 0, 0, :]
            k_out[:, 1::2, :] = d2_u8[:, :, 0, 2, :]
            v_out[:, 0::2, :] = d2_u8[:, :, 0, 1, :]
            v_out[:, 1::2, :] = d2_u8[:, :, 0, 3, :]
            return k_out, v_out

        def _u8_to_fp8(arr_u8: np.ndarray) -> jax.Array:
            return jax.lax.bitcast_convert_type(jnp.asarray(arr_u8, jnp.uint8), jnp.float8_e4m3fn)

        def _rand_fp8_u8(r_gen, shape):
            return r_gen.integers(0x28, 0x44, size=shape, dtype=np.uint8)

        rng = np.random.default_rng(1808)
        ps = 512

        # 2. dense2 + batch4/batch2 parity (odd/even RMW + evicted SWA + 0x7F tail)
        for tag_d2, mode_d2, use_env_d2, pps_d2, lens_d2 in (
            (
                "d2grp_tok_b4",
                "w8,s4,batchfull,g4k,foldcol,batch4,d2grp",
                False,
                2,
                [1, 2, 255, 511, 512, 513, 600, 750, 999, 1001, 1020, 1024, 333, 1, 1, 1] * 2,
            ),
            (
                "d2grp_env_b2",
                "w8,s3,noadapt,g4k,foldcol,batch2",
                True,
                5,
                [512, 1025, 2049, 2299, 2558, 14, 6, 1] + [1] * 24,
            ),
        ):
            with self.subTest(d2_grp=tag_d2):
                bs_d2, n_act_d2 = 32, (29 if "b4" in tag_d2 else 15)
                raw_u8 = np.full((bs_d2 * pps_d2 + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
                pi_d2 = np.arange(bs_d2 * pps_d2, dtype=np.int32).reshape(bs_d2, pps_d2)
                for s_i in range(n_act_d2):
                    old_l = lens_d2[s_i] - 1
                    for p_o in range((old_l + ps - 1) // ps):
                        pg = int(pi_d2[s_i, p_o])
                        rh = min(ps, old_l - p_o * ps)
                        raw_u8[pg, :rh, 0, :, :] = 0
                        raw_u8[pg, :rh, 0, 0, :] = _rand_fp8_u8(rng, (rh, 128))
                        raw_u8[pg, :rh, 0, 1, :] = _rand_fp8_u8(rng, (rh, 128))
                kvc_ref = _u8_to_fp8(raw_u8)
                kvc_d2 = _u8_to_fp8(_pack_dense2_u8(raw_u8))
                q_d2 = jnp.asarray((rng.standard_normal((bs_d2, 16, 128)) * 0.35).astype(np.float32), jnp.bfloat16)
                k_d2 = _u8_to_fp8(_rand_fp8_u8(rng, (bs_d2, 1, 128)))
                v_d2 = _u8_to_fp8(_rand_fp8_u8(rng, (bs_d2, 1, 128)))
                pi_j = jnp.asarray(pi_d2.ravel(), jnp.int32)
                cuq_j = jnp.arange(bs_d2 + 1, dtype=jnp.int32)
                cukv_j = jnp.arange(bs_d2 + 1, dtype=jnp.int32) * (pps_d2 * ps)
                dist_j = jnp.asarray([n_act_d2, n_act_d2, n_act_d2], jnp.int32)
                kvl_j = jnp.asarray(lens_d2, jnp.int32)
                gate_d2 = jnp.asarray((rng.standard_normal((bs_d2, 1024)) * 0.35).astype(np.float32), jnp.bfloat16)

                o_ref, c_ref = run_decode_fast_rpa(
                    q_d2, k_d2, v_d2, kvc_ref, kvl_j, pi_j, cuq_j, cukv_j, dist_j,
                    sm_scale=128.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                    mode_str=mode_d2, decode_query_size=1, fused_out_pack=2,
                    gate=gate_d2, out_norm_eps=1e-5, kv_pack_dense2=False, interpret=True,
                )
                ctx = patch.dict(os.environ, {"SGLANG_TPU_RPA_V3_DENSE2_GROUP": "1" if use_env_d2 else "0"})
                with ctx:
                    cfg_d2 = parse_decode_fast_config(
                        mode_d2, sliding_window=2049, pages_per_seq=pps_d2, page_size=ps,
                        decode_query_size=1, kv_pack_dense2=True,
                    )
                    self.assertTrue(
                        can_use_swabatch_group_runtime(
                            cfg_d2, one_block_per_seq=True, decode_query_size=1,
                            kv_pack_dense2=True, probe=None, b_tokens=bs_d2, max_num_seqs=bs_d2,
                            bkv_slot_pages=cfg_d2.swabatch_group_size * cfg_d2.max_ring_pages,
                            has_init_sv=True,
                        )
                    )
                    o_d2, c_d2 = run_decode_fast_rpa(
                        q_d2, k_d2, v_d2, kvc_d2, kvl_j, pi_j, cuq_j, cukv_j, dist_j,
                        sm_scale=128.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                        mode_str=mode_d2, decode_query_size=1, fused_out_pack=2,
                        gate=gate_d2, out_norm_eps=1e-5, kv_pack_dense2=True, interpret=True,
                    )
                a_r = np.asarray(o_ref[:n_act_d2], np.float32)
                a_d = np.asarray(o_d2[:n_act_d2], np.float32)
                self.assertTrue(np.all(np.isfinite(np.asarray(o_d2, np.float32))))
                num = float(np.sum(a_r * a_d))
                den = float(np.linalg.norm(a_r) * np.linalg.norm(a_d) + 1e-12)
                self.assertGreater(num / den, 0.99995, tag_d2)
                self.assertLessEqual(float(np.max(np.abs(a_r - a_d))), 2e-2, tag_d2)
                u_ref = np.asarray(c_ref).view(np.uint8)
                k_d2_u, v_d2_u = _unpack_dense2_u8(np.asarray(c_d2).view(np.uint8))
                for s_i in range(n_act_d2):
                    for tok in range(lens_d2[s_i]):
                        pg, off = int(pi_d2[s_i, tok // ps]), tok % ps
                        self.assertTrue(np.array_equal(k_d2_u[pg, off], u_ref[pg, off, 0, 0, :]), tag_d2)
                        self.assertTrue(np.array_equal(v_d2_u[pg, off], u_ref[pg, off, 0, 1, :]), tag_d2)

        # 3. Step (c) prestart + skip_kv_write with dense2 + batch4 under race/NaN checks
        cfg_sc = parse_decode_fast_config(
            "w8,s4,batchfull,g4k,foldcol,batch4,d2grp",
            sliding_window=2049, pages_per_seq=2, page_size=ps, decode_query_size=1, kv_pack_dense2=True,
        )
        bs_sc, pps_sc, n_act_sc = 32, 2, 31
        lens_sc = [100, 255, 512, 513, 700, 999, 1023, 1024] * 4
        raw_sc = np.full((bs_sc * pps_sc + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        pi_sc = np.arange(bs_sc * pps_sc, dtype=np.int32).reshape(bs_sc, pps_sc)
        for s_i, cl in enumerate(lens_sc):
            for p_o in range(((cl - 1) + ps - 1) // ps):
                pg = int(pi_sc[s_i, p_o])
                rh = min(ps, (cl - 1) - p_o * ps)
                raw_sc[pg, :rh, 0, :, :] = 0
                raw_sc[pg, :rh, 0, 0, :] = _rand_fp8_u8(rng, (rh, 128))
                raw_sc[pg, :rh, 0, 1, :] = _rand_fp8_u8(rng, (rh, 128))
        kvc_sc = _u8_to_fp8(_pack_dense2_u8(raw_sc))
        q_sc = jnp.asarray((rng.standard_normal((bs_sc, 16, 128)) * 0.35).astype(np.float32), jnp.bfloat16)
        k_sc = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sc, 1, 128)))
        v_sc = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sc, 1, 128)))
        kv_sc_p = jnp.pad(jnp.concatenate([k_sc, v_sc], axis=1), ((0, 0), (0, 2), (0, 0))).reshape(bs_sc, 1, 4, 128)
        gate_sc = jnp.asarray((rng.standard_normal((bs_sc, 1024)) * 0.35).astype(np.float32), jnp.bfloat16)
        pi_scj = jnp.asarray(pi_sc.ravel(), jnp.int32)
        cuq_scj = jnp.arange(bs_sc + 1, dtype=jnp.int32)
        cukv_scj = jnp.arange(bs_sc + 1, dtype=jnp.int32) * (pps_sc * ps)
        dist_scj = jnp.asarray([n_act_sc, n_act_sc, n_act_sc], jnp.int32)
        kvl_scj = jnp.asarray(lens_sc, jnp.int32)
        o_std_sc, _ = run_decode_fast_rpa(
            q_sc, k_sc, v_sc, kvc_sc, kvl_scj, pi_scj, cuq_scj, cukv_scj, dist_scj,
            sm_scale=128.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
            mode_str="w8,s4,batchfull,g4k,foldcol,batch4,d2grp", decode_query_size=1,
            fused_out_pack=2, gate=gate_sc, out_norm_eps=1e-5, kv_pack_dense2=True, interpret=True,
        )

        def _stepc_d2(klr, pir, cqr, ckvr, dr, q_r, kv_r, g_r, chbm, oout, upd, bkv_r, st_r, o_r, init_r, sm_r, s1, s2):
            pre_st = prepare_and_start_rpad_dmas(
                kv_lens_ref=klr, page_indices_ref=pir, cu_q_lens_ref=cqr, cu_kv_lens_ref=ckvr,
                distribution_ref=dr, kv_cache_hbm_ref=chbm, bkv_vmem_ref=bkv_r, bkv_sems=s1,
                sliding_window=2049, cfg=cfg_sc, one_block_per_seq=True, decode_query_size=1,
                kv_pack_dense2=True, interpret_mode=True, b_tokens=bs_sc,
                init_sv_vmem_ref=init_r, use_grp_vec=True,
            )
            _run_rpad_vmem_core(
                kv_lens_ref=klr, page_indices_ref=pir, cu_q_lens_ref=cqr, cu_kv_lens_ref=ckvr,
                distribution_ref=dr, q_vmem_ref=q_r, kv_merged_vmem_ref=kv_r, gate_vmem_ref=g_r,
                kv_cache_hbm_ref=chbm, o_out_vmem_ref=oout, updated_kv_cache_hbm_ref=upd,
                bkv_vmem_ref=bkv_r, state_vmem_ref=st_r, o_vmem_ref=o_r, kv_write_vmem_ref=None,
                init_sv_vmem_ref=init_r, written_cnt_smem_ref=sm_r, bkv_sems=s1, cache_write_sem=s2,
                sm_scale=128.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0, cfg=cfg_sc,
                one_block_per_seq=True, decode_query_size=1, in_kernel_pack2=True, has_gate=True,
                out_norm_eps=1e-5, kv_pack_dense2=True, interpret_mode=True,
                prestarted_bkv_state=pre_st, use_grp_vec=True, skip_kv_write=True,
            )

        o_pre_sc, _ = pl.pallas_call(
            _stepc_d2,
            out_shape=[jax.ShapeDtypeStruct((bs_sc, 1024), jnp.bfloat16), jax.ShapeDtypeStruct(kvc_sc.shape, kvc_sc.dtype)],
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=5,
                in_specs=[
                    pl.BlockSpec(q_sc.shape, lambda *_: (0, 0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(kv_sc_p.shape, lambda *_: (0, 0, 0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(gate_sc.shape, lambda *_: (0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                out_specs=[
                    pl.BlockSpec((bs_sc, 1024), lambda *_: (0, 0), memory_space=pltpu.VMEM),
                    pl.BlockSpec(memory_space=pltpu.HBM),
                ],
                grid=(1,),
                scratch_shapes=[
                    pltpu.VMEM((cfg_sc.num_slots, 4 * cfg_sc.max_ring_pages, ps // 2, 1, 4, 128), kvc_sc.dtype),
                    pltpu.VMEM((3, 16, 128), jnp.float32),
                    pltpu.VMEM((bs_sc, 16, 128), jnp.bfloat16),
                    pltpu.VMEM((2, 32, 16, 128), jnp.float32),
                    pltpu.SMEM((2 + cfg_sc.num_slots,), jnp.int32),
                    pltpu.SemaphoreType.DMA((cfg_sc.num_slots,)),
                    pltpu.SemaphoreType.DMA((1,)),
                ],
            ),
            input_output_aliases={8: 1},
            interpret=pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan"),
        )(kvl_scj, pi_scj, cuq_scj, cukv_scj, dist_scj, q_sc, kv_sc_p, gate_sc, kvc_sc)
        self.assertTrue(np.array_equal(np.asarray(o_pre_sc).view(np.uint16), np.asarray(o_std_sc).view(np.uint16)))

        # 4. SGLANG_TPU_RPA_V3_SHARED_HOISTKV parity on shared16 LC Full
        sh_mode = "s4,c1024,shared16,q256,shsplit,pbf16,batchfull,foldcol,shfp8"
        self.assertFalse(
            parse_decode_fast_config(sh_mode, sliding_window=None, pages_per_seq=4, page_size=ps, shared_group=16).hoist_new_kv
        )
        with patch.dict(os.environ, {"SGLANG_TPU_RPA_V3_SHARED_HOISTKV": "1"}):
            self.assertTrue(
                parse_decode_fast_config(sh_mode, sliding_window=None, pages_per_seq=4, page_size=ps, shared_group=16).hoist_new_kv
            )
            self.assertFalse(
                parse_decode_fast_config(f"{sh_mode},nohoistkv", sliding_window=None, pages_per_seq=4, page_size=ps, shared_group=16).hoist_new_kv
            )
        bs_sh, n_act_sh, pps_sh, sh_n = 16, 14, 4, 2
        sh_raw = np.full((bs_sh * pps_sh + 4, ps, 1, 4, 128), 0x7F, dtype=np.uint8)
        pi_sh = np.arange(bs_sh * pps_sh, dtype=np.int32).reshape(bs_sh, pps_sh)
        sh_pgs = np.arange(sh_n, dtype=np.int32) + bs_sh * pps_sh
        pi_sh[:, :sh_n] = sh_pgs[None, :]
        sh_raw[sh_pgs, :, 0, :, :] = 0
        sh_raw[sh_pgs, :, 0, 0, :] = _rand_fp8_u8(rng, (sh_n, ps, 128))
        sh_raw[sh_pgs, :, 0, 1, :] = _rand_fp8_u8(rng, (sh_n, ps, 128))
        lens_sh = [1500, 1503, 1511, 1601, 1700, 1800, 1900, 2001, 1400, 1300, 1200, 1100, 1025, 1026, 1, 1]
        for s_i in range(n_act_sh):
            old_l = lens_sh[s_i] - 1
            for p_o in range(sh_n, (old_l + ps - 1) // ps):
                pg = int(pi_sh[s_i, p_o])
                rh = min(ps, old_l - p_o * ps)
                sh_raw[pg, :rh, 0, :, :] = 0
                sh_raw[pg, :rh, 0, 0, :] = _rand_fp8_u8(rng, (rh, 128))
                sh_raw[pg, :rh, 0, 1, :] = _rand_fp8_u8(rng, (rh, 128))
        kvc_sh_d2 = _u8_to_fp8(_pack_dense2_u8(sh_raw))
        q_sh = jnp.asarray((rng.standard_normal((bs_sh, 16, 128)) * 0.35).astype(np.float32), jnp.bfloat16)
        k_sh = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sh, 1, 128)))
        v_sh = _u8_to_fp8(_rand_fp8_u8(rng, (bs_sh, 1, 128)))
        pi_shj = jnp.asarray(pi_sh.ravel(), jnp.int32)
        cuq_shj = jnp.arange(bs_sh + 1, dtype=jnp.int32)
        cukv_shj = jnp.arange(bs_sh + 1, dtype=jnp.int32) * (pps_sh * ps)
        dist_shj = jnp.asarray([n_act_sh, n_act_sh, n_act_sh, sh_n], jnp.int32)
        kvl_shj = jnp.asarray(lens_sh, jnp.int32)
        sh_arg = jnp.asarray([sh_n], jnp.int32)
        o_sh_no, c_sh_no = run_decode_fast_rpa(
            q_sh, k_sh, v_sh, kvc_sh_d2, kvl_shj, pi_shj, cuq_shj, cukv_shj, dist_shj,
            sm_scale=128.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
            mode_str=sh_mode, decode_query_size=1, shared_pages=sh_arg, kv_pack_dense2=True, interpret=True,
        )
        with patch.dict(os.environ, {"SGLANG_TPU_RPA_V3_SHARED_HOISTKV": "1"}):
            o_sh_hk, c_sh_hk = run_decode_fast_rpa(
                q_sh, k_sh, v_sh, kvc_sh_d2, kvl_shj, pi_shj, cuq_shj, cukv_shj, dist_shj,
                sm_scale=128.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=sh_mode, decode_query_size=1, shared_pages=sh_arg, kv_pack_dense2=True, interpret=True,
            )
        self.assertTrue(np.array_equal(np.asarray(o_sh_hk).view(np.uint16), np.asarray(o_sh_no).view(np.uint16)))
        self.assertTrue(np.array_equal(np.asarray(c_sh_hk).view(np.uint8), np.asarray(c_sh_no).view(np.uint8)))

        # 5. LC SWA Step (c) plan check (sw=2049 allowed; shared Full without group disallowed; non-shared Full clamps to ring4k)
        lc_swa_mode = "s4,c4096,shared16,q256,shsplit,pbf16,batchfull,foldcol,batch4,d2grp"
        _, _, _, grp_u, ok_u = resolve_step_c_rpa_plan(
            lc_swa_mode, sliding_window=2049, pages_per_seq=5, phys_page_rows=512,
            kv_pack_dense2=False, b_tokens=64, max_num_seqs=64,
        )
        self.assertTrue(ok_u and grp_u)
        _, _, _, grp_d, ok_d = resolve_step_c_rpa_plan(
            lc_swa_mode, sliding_window=2049, pages_per_seq=5, phys_page_rows=256,
            kv_pack_dense2=True, b_tokens=64, max_num_seqs=64,
        )
        self.assertTrue(ok_d and grp_d)
        _, _, _, _, ok_full_sh = resolve_step_c_rpa_plan(
            lc_swa_mode, sliding_window=None, pages_per_seq=64, phys_page_rows=512,
            kv_pack_dense2=False, b_tokens=64, max_num_seqs=64, has_shared_prefix=True,
        )
        self.assertFalse(ok_full_sh)
        cfg_full, _, _, _, ok_full = resolve_step_c_rpa_plan(
            lc_swa_mode, sliding_window=None, pages_per_seq=64, phys_page_rows=512,
            kv_pack_dense2=False, b_tokens=64, max_num_seqs=64,
        )
        self.assertTrue(ok_full and cfg_full.loop_kind == "ring4k")

        # 6. TPU v7 AOT exports for c256 (bs=128 d2+b4, bs=256 bd4+b4) & shared64+hoistkv
        try:
            from jax._src import tpu_info  # noqa: F401
        except ImportError:
            self.skipTest("jax._src.tpu_info not available")
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch("sglang_tpu.pallas_export.get_torch_tpu_device_kind", return_value="TPU v7"),
            tpu_info_for_cpu_export(),
        ):
            with patch.dict(os.environ, {"SGLANG_TPU_RPA_V3_DENSE2_GROUP": "1"}):
                hlo_c256_d2 = jax_export.export(
                    jax.jit(
                        lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                            q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                            sm_scale=128.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                            mode_str="w8,s4,batchfull,g4k,foldcol,batch4",
                            decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                            gate=g_, out_norm_eps=1e-5, kv_pack_dense2=True,
                        )
                    ),
                    platforms=["tpu"],
                )(
                    jax.ShapeDtypeStruct((1, 128, 8, 2, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((128, 1, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((128, 1024), jnp.bfloat16),
                    jax.ShapeDtypeStruct((264, 256, 1, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((128,), jnp.int32),
                    jax.ShapeDtypeStruct((256,), jnp.int32),
                    jax.ShapeDtypeStruct((129,), jnp.int32),
                    jax.ShapeDtypeStruct((129,), jnp.int32),
                    jax.ShapeDtypeStruct((3,), jnp.int32),
                ).mlir_module()
                self.assertIn("RPAd_fast-s4-c2048-swabatch-mp5-batchfull-b4-d2", hlo_c256_d2)
            hlo_c256_bd4 = jax_export.export(
                jax.jit(
                    lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=64.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                        mode_str="w8,s4,batchfull,g4k,foldcol,batch4",
                        decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                        gate=g_, out_norm_eps=1e-5, kv_pack_bd4=True,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, 256, 2, 4, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((256, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((256, 512), jnp.bfloat16),
                jax.ShapeDtypeStruct((520, 128, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((256,), jnp.int32),
                jax.ShapeDtypeStruct((512,), jnp.int32),
                jax.ShapeDtypeStruct((257,), jnp.int32),
                jax.ShapeDtypeStruct((257,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertIn("-b4-bd4", hlo_c256_bd4)
            with patch.dict(os.environ, {"SGLANG_TPU_RPA_V3_SHARED_HOISTKV": "1"}):
                hlo_sh_hkv = jax_export.export(
                    jax.jit(
                        lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_, sp_: run_decode_fast_rpa(
                            q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                            sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                            mode_str="s4,c2048,shared64,q1024,shsplit,pbf16,batchfull,foldcol,shfp8",
                            decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                            gate=g_, out_norm_eps=1e-5, shared_pages=sp_, kv_pack_bd4=True,
                        )
                    ),
                    platforms=["tpu"],
                )(
                    jax.ShapeDtypeStruct((1, 64, 2, 4, 128), jnp.bfloat16),
                    jax.ShapeDtypeStruct((64, 1, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((64, 512), jnp.bfloat16),
                    jax.ShapeDtypeStruct((64 * 256 + 8, 128, 1, 4, 128), jnp.float8_e4m3fn),
                    jax.ShapeDtypeStruct((64,), jnp.int32),
                    jax.ShapeDtypeStruct((64 * 256,), jnp.int32),
                    jax.ShapeDtypeStruct((65,), jnp.int32),
                    jax.ShapeDtypeStruct((65,), jnp.int32),
                    jax.ShapeDtypeStruct((3,), jnp.int32),
                    jax.ShapeDtypeStruct((1,), jnp.int32),
                ).mlir_module()
                self.assertIn("-shared64-", hlo_sh_hkv)
                self.assertIn("-hkv-bd4", hlo_sh_hkv)

            # 7. B < 32 swabatch group gate (#4381): B=16 stays unbatched, B=32 uses -b4
            cfg_minb = parse_decode_fast_config(
                "w8,s4,batchfull,g4k,foldcol,batch4",
                sliding_window=2049, pages_per_seq=2, page_size=ps, decode_query_size=1,
            )
            self.assertFalse(
                can_use_swabatch_group_runtime(
                    cfg_minb, one_block_per_seq=True, decode_query_size=1,
                    kv_pack_dense2=False, probe=None, b_tokens=16, max_num_seqs=16,
                    bkv_slot_pages=cfg_minb.swabatch_group_size * cfg_minb.max_ring_pages,
                    has_init_sv=True,
                )
            )
            self.assertTrue(
                can_use_swabatch_group_runtime(
                    cfg_minb, one_block_per_seq=True, decode_query_size=1,
                    kv_pack_dense2=False, probe=None, b_tokens=32, max_num_seqs=32,
                    bkv_slot_pages=cfg_minb.swabatch_group_size * cfg_minb.max_ring_pages,
                    has_init_sv=True,
                )
            )
            hlo_b16 = jax_export.export(
                jax.jit(
                    lambda q_, kv_, g_, c_, kl_, pi_, cq_, ckv_, d_: run_decode_fast_rpa(
                        q_, kv_, None, c_, kl_, pi_, cq_, ckv_, d_,
                        sm_scale=128.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                        mode_str="w8,s4,batchfull,g4k,foldcol,batch4",
                        decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                        gate=g_, out_norm_eps=1e-5,
                    )
                ),
                platforms=["tpu"],
            )(
                jax.ShapeDtypeStruct((1, 16, 8, 2, 128), jnp.bfloat16),
                jax.ShapeDtypeStruct((16, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((16, 1024), jnp.bfloat16),
                jax.ShapeDtypeStruct((40, 512, 1, 4, 128), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((16,), jnp.int32),
                jax.ShapeDtypeStruct((32,), jnp.int32),
                jax.ShapeDtypeStruct((17,), jnp.int32),
                jax.ShapeDtypeStruct((17,), jnp.int32),
                jax.ShapeDtypeStruct((3,), jnp.int32),
            ).mlir_module()
            self.assertNotIn("-b4", hlo_b16)
            self.assertNotIn("-b2", hlo_b16)

    def test_shared_tight_private_tail_widths_and_0x7f_poison_parity(self):
        """Verify tight 512/1024/2048/4096 private-tail dispatch + noshsplit under 0x7F poison & TPU7x AOT."""
        import gc
        from jax.experimental import topologies
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        ip_race = pltpu.InterpretParams(
            dma_execution_mode="eager",
            detect_races=True,
            uninitialized_memory="nan",
        )
        rng = np.random.default_rng(20262500)
        # 16 seqs sharing 8 pages (4096 tokens -> 1 full shared block) with diverse private tails
        # spanning every tight width tier: 0 (shared-only), 120 (<=512), 700 (<=1024), 1500 (<=2048), 2600 (<=4096).
        B, group, pps, sh_pgs = 16, 16, 16, 8
        priv_lens = [0, 120, 512, 700, 1024, 1500, 2048, 2600] * 2
        for dense2, bd4 in ((True, False), (False, True)):
            pack_factor = 4 if bd4 else 2
            phys_rows = 512 // pack_factor
            priv_pgs = pps - sh_pgs
            nb = sh_pgs + B * priv_pgs + 4
            kvc_u8 = np.full((nb, phys_rows, 1, 4, 128), 0x7F, dtype=np.uint8)
            pg_mat = np.zeros((B, pps), dtype=np.int32)
            pg_mat[:, :sh_pgs] = np.arange(sh_pgs, dtype=np.int32)
            # Fill shared pages cleanly.
            kvc_u8[:sh_pgs] = rng.integers(0x30, 0x3D, size=(sh_pgs, phys_rows, 1, 4, 128), dtype=np.uint8)
            nxt = sh_pgs
            for i in range(B):
                pg_mat[i, sh_pgs:] = np.arange(nxt, nxt + priv_pgs, dtype=np.int32)
                n_valid_priv_pgs = (priv_lens[i] + 511) // 512
                if n_valid_priv_pgs > 0:
                    # Leave unused tokens beyond priv_lens[i] as 0x7F poison to prove valid_tokens masking holds.
                    for p_rel in range(n_valid_priv_pgs):
                        pg_id = nxt + p_rel
                        toks_in_pg = min(512, priv_lens[i] - p_rel * 512)
                        n_full_phys = toks_in_pg // pack_factor
                        if n_full_phys > 0:
                            kvc_u8[pg_id, :n_full_phys] = rng.integers(
                                0x30, 0x3D, size=(n_full_phys, 1, 4, 128), dtype=np.uint8
                            )
                        rem = toks_in_pg % pack_factor
                        if rem > 0:
                            for s_i in range(rem):
                                if bd4:
                                    c0 = 0 if s_i < 2 else 2
                                    l0, l1 = (0, 64) if (s_i & 1) == 0 else (64, 128)
                                    kvc_u8[pg_id, n_full_phys, 0, c0 : c0 + 2, l0:l1] = rng.integers(
                                        0x30, 0x3D, size=(2, 64), dtype=np.uint8
                                    )
                                else:
                                    c0 = 0 if s_i == 0 else 2
                                    kvc_u8[pg_id, n_full_phys, 0, c0 : c0 + 2, :] = rng.integers(
                                        0x30, 0x3D, size=(2, 128), dtype=np.uint8
                                    )
                nxt += priv_pgs

            kvc = jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_5d = jnp.asarray(rng.standard_normal((1, B, 8, 2, 128)) * 0.1, jnp.bfloat16)
            kvm = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gate = jnp.asarray(rng.standard_normal((B, 512 if bd4 else 1024)) * 0.1, jnp.bfloat16)
            # Add +1 for the newly written decode token at kv_len - 1.
            kvl_np = np.array([sh_pgs * 512 + p + 1 for p in priv_lens], dtype=np.int32)
            kvl = jnp.asarray(kvl_np, jnp.int32)
            pg = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
            cuq = jnp.arange(B + 1, dtype=jnp.int32)
            cukv = jnp.arange(B + 1, dtype=jnp.int32) * (pps * 512)
            dist = jnp.asarray([B, B, B], jnp.int32)
            sh_p = jnp.asarray([sh_pgs], dtype=jnp.int32)

            ref_out, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=f"w8,s4,c4096,shared{group},coal,noz,pbf16,q1024,batchfull,foldcol",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                shared_pages=sh_p, interpret=True,
            )
            fast_out, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=f"w8,s4,c4096,shared{group},coal,noz,pbf16,q1024,shsplit,shfp8,batchfull,foldcol",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                shared_pages=sh_p, interpret=ip_race,
            )
            r_np = np.asarray(ref_out, dtype=np.float64).ravel()
            f_np = np.asarray(fast_out, dtype=np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(f_np)))
            cos = float(np.dot(r_np, f_np) / (np.linalg.norm(r_np) * np.linalg.norm(f_np) + 1e-12))
            self.assertGreater(cos, 0.9985, f"dense2={dense2} bd4={bd4} cos={cos}")

            # Verify TPU7x AOT compile < 63 MiB VMEM for full shared64 c4096 DENSE2 & BD4.
            tpu_devs = topologies.get_topology_desc("tpu7x:2x2x2" if bd4 else "tpu7x:2x2x1", platform="tpu").devices
            mesh = Mesh(np.array(tpu_devs[:1]), ("x",))
            s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
            abs_in = (
                s((1, 64, 8, 2, 128), jnp.bfloat16),
                s((64, 1, 4, 128), jnp.float8_e4m3fn),
                s((64, 512 if bd4 else 1024), jnp.bfloat16),
                s((64 * 512 + 8, phys_rows, 1, 4, 128), jnp.float8_e4m3fn),
                s((64,), jnp.int32),
                s((64 * 512,), jnp.int32),
                s((65,), jnp.int32),
                s((65,), jnp.int32),
                s((3,), jnp.int32),
                s((1,), jnp.int32),
            )
            with patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
                comp = jax.jit(
                    jax.shard_map(
                        lambda q_, km_, gt_, c_, kl_, pi_, cq_, ckv_, dt_, sp_: run_decode_fast_rpa(
                            q_, km_, None, c_, kl_, pi_, cq_, ckv_, dt_,
                            sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                            mode_str="w8,s4,c4096,shared64,coal,noz,pbf16,q1024,shsplit,shfp8,batchfull,foldcol",
                            decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                            gate=gt_, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                            shared_pages=sp_,
                        ),
                        mesh=mesh,
                        in_specs=(P(),) * 10,
                        out_specs=(P(), P()),
                        check_vma=False,
                    )
                ).lower(*abs_in).compile()
            self.assertIsNotNone(comp)
            del comp
            jax.clear_caches()
            gc.collect()

    def test_wave15_ring8k_tail4k_and_w7auto_oneshot8k_0x7f_poison_and_aot(self):
        """Verify ring8k 4096 tight tail, direct ring4k DMA, w7auto g4k 8K one-shot, 0x7F poison, & TPU7x AOT."""
        import gc
        from jax.experimental import topologies
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        ip_race = pltpu.InterpretParams(
            dma_execution_mode="eager",
            detect_races=True,
            uninitialized_memory="nan",
        )
        rng = np.random.default_rng(20262600)
        # 8 seqs spanning ring8k tight tiers (<=512, <=1024, <=2048, <=4096, 4096..8192, 8192, >8192) & w7auto 8K.
        B, pps = 8, 22
        kvl_list = [350, 1500, 3500, 5500, 7500, 8100, 8193, 9500]
        for dense2, bd4 in ((False, True), (True, False)):
            pack_factor = 4 if bd4 else 2
            phys_rows = 512 // pack_factor
            nb = B * pps + 4
            kvc_u8 = np.full((nb, phys_rows, 1, 4, 128), 0x7F, dtype=np.uint8)
            pg_mat = np.zeros((B, pps), dtype=np.int32)
            nxt = 0
            for i, kvl_v in enumerate(kvl_list):
                pg_mat[i] = np.arange(nxt, nxt + pps, dtype=np.int32)
                old_toks = kvl_v - 1
                n_pg = (old_toks + 511) // 512
                for p_rel in range(n_pg):
                    pg_id = nxt + p_rel
                    toks_in_pg = min(512, old_toks - p_rel * 512)
                    n_full = toks_in_pg // pack_factor
                    if n_full > 0:
                        kvc_u8[pg_id, :n_full] = rng.integers(
                            0x30, 0x3D, size=(n_full, 1, 4, 128), dtype=np.uint8
                        )
                    rem = toks_in_pg % pack_factor
                    if rem > 0:
                        for s_i in range(rem):
                            if bd4:
                                c0 = 0 if s_i < 2 else 2
                                l0, l1 = (0, 64) if (s_i & 1) == 0 else (64, 128)
                                kvc_u8[pg_id, n_full, 0, c0 : c0 + 2, l0:l1] = rng.integers(
                                    0x30, 0x3D, size=(2, 64), dtype=np.uint8
                                )
                            else:
                                c0 = 0 if s_i == 0 else 2
                                kvc_u8[pg_id, n_full, 0, c0 : c0 + 2, :] = rng.integers(
                                    0x30, 0x3D, size=(2, 128), dtype=np.uint8
                                )
                nxt += pps

            kvc = jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_5d = jnp.asarray(rng.standard_normal((1, B, 8, 2, 128)) * 0.1, jnp.bfloat16)
            kvm = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gate = jnp.asarray(rng.standard_normal((B, 512 if bd4 else 1024)) * 0.1, jnp.bfloat16)
            kvl = jnp.asarray(kvl_list, dtype=jnp.int32)
            pg = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
            cuq = jnp.arange(B + 1, dtype=jnp.int32)
            cukv = jnp.arange(B + 1, dtype=jnp.int32) * (pps * 512)
            dist = jnp.asarray([B, B, B], jnp.int32)

            ref_r4k, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,s3,ring4k,c4096,pbf16,batchfull,g4k,foldcol",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                interpret=ip_race,
            )
            out_r8k, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,s4,ring8k,nocoal,pbf16,batchfull,g4k,foldcol",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                interpret=ip_race,
            )
            r_np = np.asarray(ref_r4k, dtype=np.float64).ravel()
            f_np = np.asarray(out_r8k, dtype=np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(f_np)))
            cos_r8 = float(np.dot(r_np, f_np) / (np.linalg.norm(r_np) * np.linalg.norm(f_np) + 1e-12))
            self.assertGreater(cos_r8, 0.9999, f"ring8k bd4={bd4} cos={cos_r8}")

            # Verify w7auto g4k 8K one-shot (5120 < rel_end <= 8192) vs dyn2ku under 0x7F poison.
            kvl_w7 = jnp.asarray([350, 1000, 3500, 5500, 7500, 8100, 8193, 9500], dtype=jnp.int32)
            # Seq 1 has 1500 valid tokens populated, so kvl_w7[1]=1000 cleanly tests the <=1K gear under 0x7F poison.
            ref_d2k, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl_w7, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,s4,c1024,pbf16,batchfull,dyn2ku",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                interpret=ip_race,
            )
            out_g4k, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl_w7, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,s4,c1024,pbf16,batchfull,g4k,foldcol",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                interpret=ip_race,
            )
            rd_np = np.asarray(ref_d2k, dtype=np.float64).ravel()
            fg_np = np.asarray(out_g4k, dtype=np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(fg_np)))
            cos_w7 = float(np.dot(rd_np, fg_np) / (np.linalg.norm(rd_np) * np.linalg.norm(fg_np) + 1e-12))
            self.assertGreater(cos_w7, 0.9999, f"w7_oneshot8k bd4={bd4} cos={cos_w7}")

        # Verify TPU7x AOT compile < 63 MiB VMEM for ring8k (BD4 B=256) & w7auto g4k 8K (BD4 B=256, DENSE2 B=128, 512B B=64).
        for topo, bs_a, d2_a, bd4_a, mode_a in (
            ("tpu7x:2x2x2", 256, False, True, "w8,s4,ring8k,nocoal,pbf16,batchfull,g4k,foldcol"),
            ("tpu7x:2x2x2", 256, False, True, "w8,s4,c1024,pbf16,batchfull,g4k,foldcol"),
            ("tpu7x:2x2x1", 128, True, False, "w8,c1024,pbf16,batchfull,g4k,foldcol"),
            ("tpu7x:2x2x1", 64, False, False, "w8,c1024,pbf16,batchfull,g4k,foldcol"),
        ):
            phys_r = 128 if bd4_a else (256 if d2_a else 512)
            tpu_devs = topologies.get_topology_desc(topo, platform="tpu").devices
            mesh = Mesh(np.array(tpu_devs[:1]), ("x",))
            s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
            abs_in = (
                s((1, bs_a, 8, 2, 128), jnp.bfloat16),
                s((bs_a, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a, 512 if bd4_a else 1024), jnp.bfloat16),
                s((bs_a * 32 + 8, phys_r, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a,), jnp.int32),
                s((bs_a * 32,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((3,), jnp.int32),
            )
            with patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
                comp = jax.jit(
                    jax.shard_map(
                        lambda q_, km_, gt_, c_, kl_, pi_, cq_, ckv_, dt_: run_decode_fast_rpa(
                            q_, km_, None, c_, kl_, pi_, cq_, ckv_, dt_,
                            sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                            mode_str=mode_a, decode_query_size=1, fused_out_pack=2,
                            prepared_inputs=True, gate=gt_, out_norm_eps=1e-5,
                            kv_pack_dense2=d2_a, kv_pack_bd4=bd4_a,
                        ),
                        mesh=mesh,
                        in_specs=(P(),) * 9,
                        out_specs=(P(), P()),
                        check_vma=False,
                    )
                ).lower(*abs_in).compile()
            self.assertIsNotNone(comp)
            del comp
            jax.clear_caches()
            gc.collect()

    def test_wave16_shared_sanvmem_s4_sh8_c8192_and_dyn4k_oneshot_0x7f_poison_and_aot(self):
        """Verify shared* D2 s4, shared8 c8192, priv VMEM sanitize & dyn4k 8K oneshot w/ 0x7F poison + AOT."""
        import gc
        from jax.experimental import topologies
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            parse_decode_fast_config,
            run_decode_fast_rpa,
        )

        # 1. Config routing: shared* uses num_slots=4 on D2 (3 on c8192); shared8/64 >16K w/ batchfull
        # use c8192/s3, while explicit "shc4096" and "shared8,s4,c4096" honor explicit s4/c4096.
        cfg_sh64_d2 = parse_decode_fast_config(
            "w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
            sliding_window=None,
            pages_per_seq=516,
            page_size=512,
            shared_group=64,
            kv_pack_dense2=True,
        )
        self.assertEqual((cfg_sh64_d2.num_slots, cfg_sh64_d2.chunk_sz), (3, 8192))
        cfg_sh64_d2_4k = parse_decode_fast_config(
            "w8,c4096,shc4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
            sliding_window=None,
            pages_per_seq=516,
            page_size=512,
            shared_group=64,
            kv_pack_dense2=True,
        )
        self.assertEqual((cfg_sh64_d2_4k.num_slots, cfg_sh64_d2_4k.chunk_sz), (4, 4096))

        cfg_sh8_inh = parse_decode_fast_config(
            "w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
            sliding_window=None,
            pages_per_seq=516,
            page_size=512,
            shared_group=8,
            kv_pack_dense2=True,
        )
        self.assertEqual((cfg_sh8_inh.num_slots, cfg_sh8_inh.chunk_sz), (3, 8192))

        cfg_sh8_exp = parse_decode_fast_config(
            "w8,s4,c4096,shared8,coal,noz,pbf16,shsplit,shfp8,batchfull,foldcol",
            sliding_window=None,
            pages_per_seq=512,
            page_size=512,
            shared_group=8,
            kv_pack_dense2=True,
        )
        self.assertEqual((cfg_sh8_exp.num_slots, cfg_sh8_exp.chunk_sz), (4, 4096))

        ip_race = pltpu.InterpretParams(
            dma_execution_mode="eager",
            detect_races=True,
            uninitialized_memory="nan",
        )
        rng = np.random.default_rng(20262700)

        def _poison_kvc(kvc_u8, pg_mat, kvl_np, start_pg, bd4):
            pps = pg_mat.shape[1]
            for s_i in range(pg_mat.shape[0]):
                end_tok = max(int(kvl_np[s_i]) - 1, 0)
                hi_pg = end_tok // 512
                for p_i in range(start_pg, pps):
                    if p_i > hi_pg:
                        kvc_u8[pg_mat[s_i, p_i], :, :, :, :] = 0x7F
                    elif p_i == hi_pg:
                        rem = end_tok % 512
                        for t in range(rem, 512):
                            if bd4:
                                r, pair, c0 = t // 4, (t % 4) // 2, (t % 2) * 64
                                kvc_u8[pg_mat[s_i, p_i], r, 0, pair * 2 : pair * 2 + 2, c0 : c0 + 64] = 0x7F
                            else:
                                kvc_u8[pg_mat[s_i, p_i], t // 2, 0, (t % 2) * 2 :, :] = 0x7F

        # 2. Eager race + uninitialized_memory="nan" + 0x7F parity for shared_8 priv sanitize_slot_v2 (D2 & BD4).
        B, sh_pgs, pps = 8, 1, 21
        kvl_sh_np = np.array([sh_pgs * 512 + p for p in (200, 1200, 3900, 4097, 5120, 7500, 8192, 9500)], dtype=np.int32)
        for dense2, bd4 in ((True, False), (False, True)):
            pack_factor = 4 if bd4 else 2
            phys_rows = 512 // pack_factor
            nb = sh_pgs + B * (pps - sh_pgs) + 8
            kvc_u8 = rng.integers(0x30, 0x3D, size=(nb, phys_rows, 1, 4, 128), dtype=np.uint8)
            pg_mat = np.zeros((B, pps), dtype=np.int32)
            pg_mat[:, :sh_pgs] = np.arange(sh_pgs, dtype=np.int32)
            pg_mat[:, sh_pgs:] = (sh_pgs + rng.permutation(B * (pps - sh_pgs)).astype(np.int32)).reshape(B, pps - sh_pgs)
            _poison_kvc(kvc_u8, pg_mat, kvl_sh_np, sh_pgs, bd4)
            kvc = jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_5d = jnp.asarray(rng.standard_normal((1, B, 8, 2, 128)) * 0.1, jnp.bfloat16)
            kvm = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gate = jnp.asarray(rng.standard_normal((B, 512 if bd4 else 1024)) * 0.1, jnp.bfloat16)
            kvl = jnp.asarray(kvl_sh_np, jnp.int32)
            pg = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
            cuq = jnp.arange(B + 1, dtype=jnp.int32)
            cukv = jnp.arange(B + 1, dtype=jnp.int32) * (pps * 512)
            dist = jnp.asarray([B, B, B], jnp.int32)
            sh_p = jnp.asarray([sh_pgs], jnp.int32)

            ref_r4, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist[:3],
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,s4,ring4k,noz,pbf16,batchfull,foldcol",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                interpret=ip_race,
            )
            out_sh, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,s4,c4096,shared8,noz,pbf16,q256,shfp8,batchfull,foldcol",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                shared_pages=sh_p, interpret=ip_race,
            )
            r_np = np.asarray(ref_r4, dtype=np.float64).ravel()
            f_np = np.asarray(out_sh, dtype=np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(f_np)))
            cos_sh = float(np.dot(r_np, f_np) / (np.linalg.norm(r_np) * np.linalg.norm(f_np) + 1e-12))
            self.assertGreater(cos_sh, 0.9985, f"shared8_sanvmem bd4={bd4} cos={cos_sh}")

        # 3. Eager race + uninitialized_memory="nan" + 0x7F parity for dyn4k 1K/4K/8K oneshot gears (D2 & BD4).
        kvl_d8_np = np.array([350, 1025, 2500, 4097, 6000, 8192, 8193, 9500], dtype=np.int32)
        for dense2, bd4 in ((True, False), (False, True)):
            phys_rows = 128 if bd4 else 256
            pg_mat_d = rng.permutation(B * 20).astype(np.int32).reshape(B, 20)
            kvc_d_u8 = rng.integers(0x30, 0x3D, size=(B * 20 + 4, phys_rows, 1, 4, 128), dtype=np.uint8)
            _poison_kvc(kvc_d_u8, pg_mat_d, kvl_d8_np, 0, bd4)
            kvc_d = jax.lax.bitcast_convert_type(jnp.asarray(kvc_d_u8, jnp.uint8), jnp.float8_e4m3fn)
            gate_d = jnp.asarray(rng.standard_normal((B, 512 if bd4 else 1024)) * 0.1, jnp.bfloat16)
            kvl_d = jnp.asarray(kvl_d8_np, jnp.int32)
            pg_d = jnp.asarray(pg_mat_d.reshape(-1), jnp.int32)
            cukv_d = jnp.arange(B + 1, dtype=jnp.int32) * (20 * 512)

            ref_d2k, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc_d, kvl_d, pg_d, cuq, cukv_d, dist[:3],
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,c1024,batchfull,dyn2k",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate_d, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                interpret=ip_race,
            )
            out_d4k, _ = run_decode_fast_rpa(
                q_5d, kvm, None, kvc_d, kvl_d, pg_d, cuq, cukv_d, dist[:3],
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str="w8,c1024,batchfull,dyn4k",
                decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate_d, out_norm_eps=1e-5, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
                interpret=ip_race,
            )
            rd = np.asarray(ref_d2k, dtype=np.float64).ravel()
            fd = np.asarray(out_d4k, dtype=np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(fd)))
            cos_d = float(np.dot(rd, fd) / (np.linalg.norm(rd) * np.linalg.norm(fd) + 1e-12))
            self.assertGreater(cos_d, 0.9999, f"dyn4k_oneshot8k bd4={bd4} cos={cos_d}")

        # 4. TPU7x AOT (<32 MiB VMEM) for shared8 c8192 s3, shared64 D2 s4, nod2 dyn4k pps=136, & B=8/64 dyn4k p20.
        aot_cases = [
            ("tpu7x:2x2x1", 64, 520, True, False, True, "w8,c4096,shared64,noz,pbf16,q1024,shfp8,batchfull,foldcol"),
            ("tpu7x:2x2x1", 8, 520, True, False, True, "w8,c4096,shared64,noz,pbf16,q1024,shfp8,batchfull,foldcol"),
            ("tpu7x:2x2x2", 8, 520, False, True, True, "tp16,w8,c4096,shared64,noz,pbf16,q256,shfp8,batchfull,foldcol"),
            ("tpu7x:2x2x1", 64, 136, False, False, False, "w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x1", 8, 20, True, False, False, "w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x2", 8, 20, False, True, False, "tp16,w8,c1024,batchfull,dyn4k"),
        ]
        for topo, bs_a, pps_a, d2_a, bd4_a, sh_a, mode_a in aot_cases:
            phys_r = 128 if bd4_a else (256 if d2_a else 512)
            tpu_devs = topologies.get_topology_desc(topo, platform="tpu").devices
            mesh = Mesh(np.array(tpu_devs[:1]), ("x",))
            s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
            abs_in = (
                s((1, bs_a, 8, 2, 128), jnp.bfloat16),
                s((bs_a, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a, 512 if bd4_a else 1024), jnp.bfloat16),
                s((bs_a * pps_a + 8, phys_r, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a,), jnp.int32),
                s((bs_a * pps_a,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((4 if sh_a else 3,), jnp.int32),
                s((1,), jnp.int32),
            )
            with patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
                comp = jax.jit(
                    jax.shard_map(
                        lambda q_, km_, gt_, c_, kl_, pi_, cq_, ckv_, dt_, sp_, m=mode_a, d2=d2_a, b4=bd4_a, us=sh_a: run_decode_fast_rpa(
                            q_, km_, None, c_, kl_, pi_, cq_, ckv_, dt_,
                            sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                            mode_str=m, decode_query_size=1, fused_out_pack=2,
                            prepared_inputs=True, gate=gt_, out_norm_eps=1e-5,
                            kv_pack_dense2=d2, kv_pack_bd4=b4,
                            shared_pages=sp_ if us else None,
                        ),
                        mesh=mesh,
                        in_specs=(P(),) * 10,
                        out_specs=(P(), P()),
                        check_vma=False,
                    )
                ).lower(*abs_in).compile()
            self.assertIsNotNone(comp)
            del comp
            jax.clear_caches()
            gc.collect()

    def test_wave17_ring4k_group_and_shared_priv_grp_parity_and_aot(self):
        """Verify Wave-17 ring4k_b4, shared16/32 c8192, & shared64 priv_grp=4 eager race/0x7F/kvc parity & TPU7x AOT."""
        import gc
        from unittest.mock import patch

        import jax
        import jax.numpy as jnp
        from jax.experimental import topologies
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            parse_decode_fast_config,
            run_decode_fast_rpa,
        )

        # 1. Config routing for Wave-17 group ring4k, shared16/32 c8192, and shared64 priv_grp.
        c_256 = parse_decode_fast_config(
            "tp16,w8,c1024,batchfull,dyn4k",
            sliding_window=None, pages_per_seq=20, page_size=512, kv_pack_bd4=True, b_tokens=256,
        )
        self.assertEqual((c_256.loop_kind, c_256.num_slots, c_256.chunk_sz, c_256.ring_group_size, c_256.fold_col), ("ring4k", 3, 4096, 4, True))
        c_64_bd4 = parse_decode_fast_config(
            "tp16,w8,c1024,batchfull,dyn4k",
            sliding_window=None, pages_per_seq=20, page_size=512, kv_pack_bd4=True, b_tokens=64,
        )
        self.assertEqual((c_64_bd4.loop_kind, c_64_bd4.num_slots, c_64_bd4.chunk_sz, c_64_bd4.ring_group_size), ("ring4k", 3, 8192, 4))
        c_64_d2 = parse_decode_fast_config(
            "w8,c1024,batchfull,dyn4k",
            sliding_window=None, pages_per_seq=20, page_size=512, kv_pack_dense2=True, b_tokens=64,
        )
        self.assertEqual((c_64_d2.loop_kind, c_64_d2.num_slots, c_64_d2.chunk_sz, c_64_d2.ring_group_size), ("ring4k", 4, 4096, 4))
        c_sh16 = parse_decode_fast_config(
            "w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
            sliding_window=None, pages_per_seq=516, page_size=512, shared_group=16, kv_pack_dense2=True,
        )
        self.assertEqual((c_sh16.num_slots, c_sh16.chunk_sz, c_sh16.priv_group_size), (4, 8192, 1))
        c_sh32 = parse_decode_fast_config(
            "w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
            sliding_window=None, pages_per_seq=516, page_size=512, shared_group=32, kv_pack_dense2=True,
        )
        self.assertEqual((c_sh32.num_slots, c_sh32.chunk_sz, c_sh32.shared_q_tile, c_sh32.priv_group_size), (3, 8192, 512, 1))
        c_sh64_d2 = parse_decode_fast_config(
            "w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
            sliding_window=None, pages_per_seq=516, page_size=512, shared_group=64, kv_pack_dense2=True,
        )
        self.assertEqual((c_sh64_d2.priv_group_size, c_sh64_d2.priv_blk_pages), (4, 4))
        c_sh64_bd4 = parse_decode_fast_config(
            "tp16,w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
            sliding_window=None, pages_per_seq=516, page_size=512, shared_group=64, kv_pack_bd4=True,
        )
        self.assertEqual((c_sh64_bd4.priv_group_size, c_sh64_bd4.priv_blk_pages), (4, 8))

        ip_race = pltpu.InterpretParams(
            dma_execution_mode="eager",
            detect_races=True,
            uninitialized_memory="nan",
        )
        rng = np.random.default_rng(20262800)

        def _poison_kvc(kvc_u8, pg_mat, kvl_np, start_pg, bd4, n_act):
            pps = pg_mat.shape[1]
            for s_i in range(n_act):
                end_tok = max(int(kvl_np[s_i]) - 1, 0)
                hi_pg = end_tok // 512
                for p_i in range(start_pg, pps):
                    if p_i > hi_pg:
                        kvc_u8[pg_mat[s_i, p_i], :, :, :, :] = 0x7F
                    elif p_i == hi_pg:
                        rem = end_tok % 512
                        for t in range(rem, 512):
                            if bd4:
                                r, pair, c0 = t // 4, (t % 4) // 2, (t % 2) * 64
                                kvc_u8[pg_mat[s_i, p_i], r, 0, pair * 2 : pair * 2 + 2, c0 : c0 + 64] = 0x7F
                            else:
                                kvc_u8[pg_mat[s_i, p_i], t // 2, 0, (t % 2) * 2 :, :] = 0x7F

        # 2. Eager race + NaN + 0x7F + mixed-len + partial-batch + kvc parity for ring4k_b4 (D2 & BD4).
        B_r, n_act_r, pps_r = 32, 29, 18
        kvl_r_np = np.full((B_r,), 8600, dtype=np.int32)
        kvl_r_np[:13] = np.array(
            [1, 350, 700, 960, 1025, 2500, 4096, 4097, 5122, 6000, 7500, 8192, 8500],
            dtype=np.int32,
        )
        for d2, bd4 in ((True, False), (False, True)):
            phys_rows = 128 if bd4 else 256
            cols = 512 if bd4 else 1024
            pg_mat = rng.permutation(B_r * pps_r).astype(np.int32).reshape(B_r, pps_r)
            kvc_u8 = rng.integers(0x30, 0x3D, size=(B_r * pps_r + 4, phys_rows, 1, 4, 128), dtype=np.uint8)
            _poison_kvc(kvc_u8, pg_mat, kvl_r_np, 0, bd4, n_act_r)
            kvc = jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_5d = jnp.asarray(rng.standard_normal((1, B_r, 8, 2, 128)) * 0.1, jnp.bfloat16)
            kvm = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B_r, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gate = jnp.asarray(rng.standard_normal((B_r, cols)) * 0.1, jnp.bfloat16)
            kvl = jnp.asarray(kvl_r_np, jnp.int32)
            pg = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
            cuq = jnp.asarray(np.concatenate([np.arange(n_act_r + 1), np.full((B_r - n_act_r,), n_act_r)], dtype=np.int32))
            cukv = jnp.arange(B_r + 1, dtype=jnp.int32) * (pps_r * 512)
            dist = jnp.asarray([n_act_r, n_act_r, n_act_r], jnp.int32)
            ref_m = "tp16,w8,c1024,batchfull,dyn4k,b1" if bd4 else "w8,c1024,batchfull,dyn4k,b1"
            grp_m = "tp16,w8,c1024,batchfull,dyn4k" if bd4 else "w8,c1024,batchfull,dyn4k"
            ref_o, ref_kv = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=ref_m, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, interpret=ip_race,
            )
            grp_o, grp_kv = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=grp_m, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, interpret=ip_race,
            )
            r_np, g_np = np.asarray(ref_o, np.float64).ravel(), np.asarray(grp_o, np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(g_np)))
            cos_rg = float(np.dot(r_np, g_np) / (np.linalg.norm(r_np) * np.linalg.norm(g_np) + 1e-12))
            self.assertGreater(cos_rg, 0.9999, f"ring4k_grp bd4={bd4} cos={cos_rg}")
            self.assertTrue(np.array_equal(np.asarray(ref_kv).view(np.uint8), np.asarray(grp_kv).view(np.uint8)))

        # 3. Eager race + NaN + 0x7F + mixed-len + shared-only + kvc parity for shared64 priv_grp=4 (D2 & BD4).
        B_s, sh_pgs, pps_s = 64, 2, 12
        kvl_s_np = np.full((B_s,), sh_pgs * 512 + 3200, dtype=np.int32)
        kvl_s_np[0], kvl_s_np[1], kvl_s_np[2], kvl_s_np[3] = sh_pgs * 512, sh_pgs * 512 + 123, sh_pgs * 512 + 2051, sh_pgs * 512 + 4500
        for d2, bd4 in ((True, False), (False, True)):
            phys_rows = 128 if bd4 else 256
            cols = 512 if bd4 else 1024
            nb_s = sh_pgs + B_s * (pps_s - sh_pgs) + 8
            pg_mat_s = np.zeros((B_s, pps_s), dtype=np.int32)
            pg_mat_s[:, :sh_pgs] = np.arange(sh_pgs, dtype=np.int32)
            pg_mat_s[:, sh_pgs:] = (sh_pgs + rng.permutation(B_s * (pps_s - sh_pgs)).astype(np.int32)).reshape(B_s, pps_s - sh_pgs)
            kvc_s_u8 = rng.integers(0x30, 0x3D, size=(nb_s, phys_rows, 1, 4, 128), dtype=np.uint8)
            _poison_kvc(kvc_s_u8, pg_mat_s, kvl_s_np, sh_pgs, bd4, B_s)
            kvc_s = jax.lax.bitcast_convert_type(jnp.asarray(kvc_s_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_s5 = jnp.asarray(rng.standard_normal((1, B_s, 8, 2, 128)) * 0.1, jnp.bfloat16)
            km_s = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B_s, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gt_s = jnp.asarray(rng.standard_normal((B_s, cols)) * 0.1, jnp.bfloat16)
            kvl_s = jnp.asarray(kvl_s_np, jnp.int32)
            pg_s = jnp.asarray(pg_mat_s.reshape(-1), jnp.int32)
            cuq_s = jnp.arange(B_s + 1, dtype=jnp.int32)
            ckv_s = jnp.arange(B_s + 1, dtype=jnp.int32) * (pps_s * 512)
            dist_s = jnp.asarray([B_s, B_s, B_s], jnp.int32)
            sp_s = jnp.asarray([sh_pgs], jnp.int32)
            base_m = ("tp16," if bd4 else "") + "w8,c4096,shared64,coal,noz,pbf16,q256,shfp8,batchfull,foldcol"
            ref_so, ref_sk = run_decode_fast_rpa(
                q_s5, km_s, None, kvc_s, kvl_s, pg_s, cuq_s, ckv_s, dist_s,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=f"{base_m},noprivgrp", decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gt_s, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, shared_pages=sp_s, interpret=ip_race,
            )
            grp_so, grp_sk = run_decode_fast_rpa(
                q_s5, km_s, None, kvc_s, kvl_s, pg_s, cuq_s, ckv_s, dist_s,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=base_m, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gt_s, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, shared_pages=sp_s, interpret=ip_race,
            )
            rs_np, gs_np = np.asarray(ref_so, np.float64).ravel(), np.asarray(grp_so, np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(gs_np)))
            cos_sp = float(np.dot(rs_np, gs_np) / (np.linalg.norm(rs_np) * np.linalg.norm(gs_np) + 1e-12))
            self.assertGreater(cos_sp, 0.9999, f"shared64_priv_grp bd4={bd4} cos={cos_sp}")
            self.assertTrue(np.array_equal(np.asarray(ref_sk).view(np.uint8), np.asarray(grp_sk).view(np.uint8)))

        # 4. TPU7x AOT (<32 MiB VMEM) for Wave-17 B=256/64 ring_grp & B=16/32/64 shared*.
        aot_w17 = [
            ("tpu7x:2x2x2", 256, 20, False, True,  False, "tp16,w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x1",  64, 20, True,  False, False, "w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x2",  64, 20, False, True,  False, "tp16,w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x1",  16, 520, True, False, True,  "w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol"),
            ("tpu7x:2x2x1",  32, 520, True, False, True,  "w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol"),
            ("tpu7x:2x2x2",  64, 520, False, True, True,  "tp16,w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol"),
        ]
        for topo, bs_a, pps_a, d2_a, bd4_a, sh_a, mode_a in aot_w17:
            phys_r = 128 if bd4_a else 256
            tpu_devs = topologies.get_topology_desc(topo, platform="tpu").devices
            mesh = Mesh(np.array(tpu_devs[:1]), ("x",))
            s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
            abs_in = (
                s((1, bs_a, 8, 2, 128), jnp.bfloat16),
                s((bs_a, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a, 512 if bd4_a else 1024), jnp.bfloat16),
                s((bs_a * pps_a + 8, phys_r, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a,), jnp.int32),
                s((bs_a * pps_a,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((4 if sh_a else 3,), jnp.int32),
                s((1,), jnp.int32),
            )
            with patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
                comp = jax.jit(
                    jax.shard_map(
                        lambda q_, km_, gt_, c_, kl_, pi_, cq_, ckv_, dt_, sp_, m=mode_a, d2=d2_a, b4=bd4_a, us=sh_a: run_decode_fast_rpa(
                            q_, km_, None, c_, kl_, pi_, cq_, ckv_, dt_,
                            sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                            mode_str=m, decode_query_size=1, fused_out_pack=2,
                            prepared_inputs=True, gate=gt_, out_norm_eps=1e-5,
                            kv_pack_dense2=d2, kv_pack_bd4=b4,
                            shared_pages=sp_ if us else None,
                        ),
                        mesh=mesh,
                        in_specs=(P(),) * 10,
                        out_specs=(P(), P()),
                        check_vma=False,
                    )
                ).lower(*abs_in).compile()
            self.assertIsNotNone(comp)
            del comp
            jax.clear_caches()
            gc.collect()

    def test_wave18_swa_group_and_shared_subchunk_and_dyn4k_guard(self):
        """Verify #5468 plain dyn4k guard, auto swabatch b4/d2grp, shared sh_sub_chunk/q512/foldcol race/NaN/0x7F parity, and TPU7x AOT."""
        import gc
        from unittest.mock import patch

        from jax.experimental import topologies
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            parse_decode_fast_config,
            run_decode_fast_rpa,
        )

        # 1. #5468: plain "dyn4k" (no w7/batchfull/b4/b2) stays streamdyn instead of ungrouped ring4k.
        cfg_plain_dyn4k = parse_decode_fast_config(
            "dyn4k",
            sliding_window=None,
            pages_per_seq=20,
            page_size=512,
            kv_pack_dense2=True,
            b_tokens=64,
        )
        self.assertEqual(cfg_plain_dyn4k.loop_kind, "streamdyn")
        self.assertEqual(cfg_plain_dyn4k.ring_group_size, 1)

        # #5581: batch4 takes precedence over batch2 in eff_ring_grp (matching swabatch_group_size).
        cfg_b2_b4 = parse_decode_fast_config(
            "w8,noadapt,hoistkv,g4k,foldcol,batch2,batch4",
            sliding_window=None,
            pages_per_seq=32,
            page_size=512,
            kv_pack_dense2=True,
            b_tokens=128,
        )
        self.assertEqual((cfg_b2_b4.loop_kind, cfg_b2_b4.ring_group_size), ("ring4k", 4))
        cfg_b2_only = parse_decode_fast_config(
            "w8,noadapt,hoistkv,g4k,foldcol,batch2",
            sliding_window=None,
            pages_per_seq=32,
            page_size=512,
            kv_pack_dense2=True,
            b_tokens=128,
        )
        self.assertEqual((cfg_b2_only.loop_kind, cfg_b2_only.ring_group_size), ("ring4k", 2))

        # Auto swabatch g8/g16 (B>=64) & g4 (32<=B<64) + d2grp + foldcol, while B<32 stays safe.
        cfg_swa_d2_64 = parse_decode_fast_config(
            "w8,c1024,batchfull,dyn4k",
            sliding_window=2049,
            pages_per_seq=8,
            page_size=512,
            kv_pack_dense2=True,
            b_tokens=64,
        )
        self.assertEqual(
            (cfg_swa_d2_64.loop_kind, cfg_swa_d2_64.swabatch_group_size, cfg_swa_d2_64.dense2_group, cfg_swa_d2_64.fold_col),
            ("swabatch", 8, True, True),
        )
        cfg_swa_d2_16 = parse_decode_fast_config(
            "w8,c1024,batchfull,dyn4k",
            sliding_window=2049,
            pages_per_seq=8,
            page_size=512,
            kv_pack_dense2=True,
            b_tokens=16,
        )
        self.assertEqual((cfg_swa_d2_16.swabatch_group_size, cfg_swa_d2_16.dense2_group), (1, False))

        # Shared c4096/c8192 auto-enables c8192+hkv (shared>=64), foldcol, q512, and sh_sub_chunk=2048 for shared>=16 (0 for shared8).
        cfg_sh64 = parse_decode_fast_config(
            "w8,c4096,shared64,shsplit,pbf16,batchfull",
            sliding_window=None,
            pages_per_seq=529,
            page_size=512,
            shared_group=64,
            kv_pack_dense2=True,
            b_tokens=64,
        )
        self.assertEqual(
            (cfg_sh64.chunk_sz, cfg_sh64.num_slots, cfg_sh64.shared_q_tile, cfg_sh64.sh_sub_chunk, cfg_sh64.fold_col, cfg_sh64.hoist_new_kv),
            (8192, 3, 512, 2048, True, True),
        )
        cfg_sh8 = parse_decode_fast_config(
            "w8,c4096,shared64,shsplit,pbf16,batchfull",
            sliding_window=None,
            pages_per_seq=529,
            page_size=512,
            shared_group=8,
            kv_pack_dense2=True,
            b_tokens=8,
        )
        self.assertEqual((cfg_sh8.chunk_sz, cfg_sh8.sh_sub_chunk, cfg_sh8.fold_col), (8192, 0, True))

        def _poison_kvc(kvc_u8, pg_mat, kvl_np, start_pg, bd4, n_act):
            pps = pg_mat.shape[1]
            for s in range(pg_mat.shape[0]):
                end_tok = max(int(kvl_np[s]) - 1, 0) if s < n_act else 0
                hi_pg = end_tok // 512
                for p in range(start_pg, pps):
                    if p > hi_pg or s >= n_act:
                        kvc_u8[pg_mat[s, p], :, :, :, :] = 0x7F
                    elif p == hi_pg:
                        for t in range(end_tok % 512, 512):
                            if bd4:
                                r, pair, c0 = t // 4, (t % 4) // 2, (t % 2) * 64
                                kvc_u8[pg_mat[s, p], r, 0, pair * 2 : pair * 2 + 2, c0 : c0 + 64] = 0x7F
                            else:
                                kvc_u8[pg_mat[s, p], t // 2, 0, (t % 2) * 2 :, :] = 0x7F

        rng = np.random.default_rng(20261801)
        ip_race = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")

        # 2. Eager race + NaN + 0x7F + mixed-len + inactive SWA parity (D2 & BD4 at B=32).
        B_sw, pps_sw, n_act_sw = 32, 6, 30
        kvl_sw = np.full((B_sw,), 2049, dtype=np.int32)
        kvl_sw[0], kvl_sw[1], kvl_sw[2], kvl_sw[3], kvl_sw[4] = 1, 350, 513, 1400, 2400
        for d2, bd4 in ((True, False), (False, True)):
            phys_rows = 128 if bd4 else 256
            cols = 512 if bd4 else 1024
            pg_mat = rng.permutation(B_sw * pps_sw).astype(np.int32).reshape(B_sw, pps_sw)
            kvc_u8 = rng.integers(0x30, 0x3D, size=(B_sw * pps_sw + 4, phys_rows, 1, 4, 128), dtype=np.uint8)
            _poison_kvc(kvc_u8, pg_mat, kvl_sw, 0, bd4, n_act_sw)
            kvc = jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_5d = jnp.asarray(rng.standard_normal((1, B_sw, 8, 2, 128)) * 0.1, jnp.bfloat16)
            kvm = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B_sw, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gate = jnp.asarray(rng.standard_normal((B_sw, cols)) * 0.1, jnp.bfloat16)
            kvl = jnp.asarray(kvl_sw, jnp.int32)
            pg = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
            cuq = jnp.asarray(np.concatenate([np.arange(n_act_sw + 1), np.full((B_sw - n_act_sw,), n_act_sw)], dtype=np.int32))
            cukv = jnp.arange(B_sw + 1, dtype=jnp.int32) * (pps_sw * 512)
            dist = jnp.asarray([n_act_sw, n_act_sw, n_act_sw], jnp.int32)
            base_m = ("tp16," if bd4 else "") + "w8,c1024,batchfull,dyn4k"
            ref_o, ref_kv = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                mode_str=f"{base_m},b1,nofold", decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, interpret=ip_race,
            )
            grp_o, grp_kv = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=2049, k_scale=1.0, v_scale=1.0,
                mode_str=base_m, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, interpret=ip_race,
            )
            r_np, g_np = np.asarray(ref_o, np.float64).ravel(), np.asarray(grp_o, np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(g_np)))
            cos_sw = float(np.dot(r_np, g_np) / (np.linalg.norm(r_np) * np.linalg.norm(g_np) + 1e-12))
            self.assertGreater(cos_sw, 0.99999, f"swa_grp bd4={bd4} cos={cos_sw}")
            self.assertTrue(np.array_equal(np.asarray(ref_kv).view(np.uint8), np.asarray(grp_kv).view(np.uint8)))

        # 3. Eager race + NaN + 0x7F shared sh_sub_chunk=2048 + q512 + foldcol parity (D2 & BD4 shared16).
        B_sh, sh_pgs, pps_sh = 16, 8, 12  # 4096 shared + up to 4 priv pages -> tests shsub2k across full 4096 block
        kvl_sh = np.full((B_sh,), sh_pgs * 512 + 1200, dtype=np.int32)
        kvl_sh[0], kvl_sh[1] = sh_pgs * 512, sh_pgs * 512 + 350
        for d2, bd4 in ((True, False), (False, True)):
            phys_rows = 128 if bd4 else 256
            cols = 512 if bd4 else 1024
            nb_s = sh_pgs + B_sh * (pps_sh - sh_pgs) + 8
            pg_mat_s = np.zeros((B_sh, pps_sh), dtype=np.int32)
            pg_mat_s[:, :sh_pgs] = np.arange(sh_pgs, dtype=np.int32)
            pg_mat_s[:, sh_pgs:] = (sh_pgs + rng.permutation(B_sh * (pps_sh - sh_pgs)).astype(np.int32)).reshape(B_sh, pps_sh - sh_pgs)
            kvc_s_u8 = rng.integers(0x30, 0x3D, size=(nb_s, phys_rows, 1, 4, 128), dtype=np.uint8)
            _poison_kvc(kvc_s_u8, pg_mat_s, kvl_sh, sh_pgs, bd4, B_sh)
            kvc_s = jax.lax.bitcast_convert_type(jnp.asarray(kvc_s_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_s5 = jnp.asarray(rng.standard_normal((1, B_sh, 8, 2, 128)) * 0.1, jnp.bfloat16)
            km_s = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B_sh, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gt_s = jnp.asarray(rng.standard_normal((B_sh, cols)) * 0.1, jnp.bfloat16)
            kvl_s = jnp.asarray(kvl_sh, jnp.int32)
            pg_s = jnp.asarray(pg_mat_s.reshape(-1), jnp.int32)
            cuq_s = jnp.arange(B_sh + 1, dtype=jnp.int32)
            ckv_s = jnp.arange(B_sh + 1, dtype=jnp.int32) * (pps_sh * 512)
            dist_s = jnp.asarray([B_sh, B_sh, B_sh], jnp.int32)
            sp_s = jnp.asarray([sh_pgs], jnp.int32)
            base_sh = ("tp16," if bd4 else "") + "w8,c4096,shared64,pbf16,batchfull"
            ref_so, ref_sk = run_decode_fast_rpa(
                q_s5, km_s, None, kvc_s, kvl_s, pg_s, cuq_s, ckv_s, dist_s,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=f"{base_sh},noshsub,nofold,forceq256", decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gt_s, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, shared_pages=sp_s, interpret=ip_race,
            )
            sub_so, sub_sk = run_decode_fast_rpa(
                q_s5, km_s, None, kvc_s, kvl_s, pg_s, cuq_s, ckv_s, dist_s,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=base_sh, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gt_s, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, shared_pages=sp_s, interpret=ip_race,
            )
            rs_np, ss_np = np.asarray(ref_so, np.float64).ravel(), np.asarray(sub_so, np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(ss_np)))
            cos_sh = float(np.dot(rs_np, ss_np) / (np.linalg.norm(rs_np) * np.linalg.norm(ss_np) + 1e-12))
            self.assertGreater(cos_sh, 0.99999, f"sh_sub_chunk bd4={bd4} cos={cos_sh}")
            self.assertTrue(np.array_equal(np.asarray(ref_sk).view(np.uint8), np.asarray(sub_sk).view(np.uint8)))

        # 3b. Eager race + NaN + 0x7F ring4k/ring8k tail_short_pg=2 parity across >8K blocks (17..18 pgs).
        B_t2, pps_t2, n_act_t2 = 32, 20, 30
        kvl_t2 = np.full((B_t2,), 8192, dtype=np.int32)
        kvl_t2[:4] = [350, 8192, 8705, 9216]   # block-0 / block-1 / rem_pg=1 / rem_pg=2
        kvl_t2[4:8] = [900, 1024, 512, 17]     # b_i==0 short_pg=2 group
        kvl_t2[8:12] = [8704, 7000, 6500, 8193]
        for d2, bd4 in ((True, False), (False, True)):
            phys_rows = 128 if bd4 else 256
            cols = 512 if bd4 else 1024
            pg_mat_t = rng.permutation(B_t2 * pps_t2).astype(np.int32).reshape(B_t2, pps_t2)
            kvc_t_u8 = rng.integers(0x30, 0x3D, size=(B_t2 * pps_t2 + 4, phys_rows, 1, 4, 128), dtype=np.uint8)
            _poison_kvc(kvc_t_u8, pg_mat_t, kvl_t2, 0, bd4, n_act_t2)
            kvc_t = jax.lax.bitcast_convert_type(jnp.asarray(kvc_t_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_t5 = jnp.asarray(rng.standard_normal((1, B_t2, 8, 2, 128)) * 0.1, jnp.bfloat16)
            km_t = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B_t2, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gt_t = jnp.asarray(rng.standard_normal((B_t2, cols)) * 0.1, jnp.bfloat16)
            kvl_tj = jnp.asarray(kvl_t2, jnp.int32)
            pg_t = jnp.asarray(pg_mat_t.reshape(-1), jnp.int32)
            cuq_t = jnp.asarray(np.concatenate([np.arange(n_act_t2 + 1), np.full((B_t2 - n_act_t2,), n_act_t2)], dtype=np.int32))
            cukv_t = jnp.arange(B_t2 + 1, dtype=jnp.int32) * (pps_t2 * 512)
            dist_t = jnp.asarray([n_act_t2, n_act_t2, n_act_t2], jnp.int32)
            base_t = ("tp16," if bd4 else "") + "w8,noadapt,hoistkv,g4k,foldcol,batch2,batch4"
            ref_to, ref_tk = run_decode_fast_rpa(
                q_t5, km_t, None, kvc_t, kvl_tj, pg_t, cuq_t, cukv_t, dist_t,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=f"{base_t},b1", decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gt_t, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, interpret=ip_race,
            )
            grp_to, grp_tk = run_decode_fast_rpa(
                q_t5, km_t, None, kvc_t, kvl_tj, pg_t, cuq_t, cukv_t, dist_t,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=base_t, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gt_t, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, interpret=ip_race,
            )
            rt_np, gt_np = np.asarray(ref_to, np.float64).ravel(), np.asarray(grp_to, np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(gt_np)))
            cos_t2 = float(np.dot(rt_np, gt_np) / (np.linalg.norm(rt_np) * np.linalg.norm(gt_np) + 1e-12))
            self.assertGreater(cos_t2, 0.99999, f"tail_short_pg2 bd4={bd4} cos={cos_t2}")
            self.assertTrue(np.array_equal(np.asarray(ref_tk).view(np.uint8), np.asarray(grp_tk).view(np.uint8)))

        # 4. TPU7x AOT (<32 MiB VMEM) for Wave-18 SWA grp4 & shared* c4096/c8192 q512 shsub2k.
        aot_w18 = [
            ("tpu7x:2x2x2", 256, 8,  False, True,  2049, False, "tp16,w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x1",  64, 8,  True,  False, 2049, False, "w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x2",  64, 8,  False, True,  2049, False, "tp16,w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x1",  64, 520, True, False, None, True,  "w8,c4096,shared64,shsplit,pbf16,batchfull"),
            ("tpu7x:2x2x2",  64, 520, False, True, None, True,  "tp16,w8,c4096,shared64,q256,pbf16,batchfull"),
            ("tpu7x:2x2x1",   8, 520, True, False, None, True,  "w8,c4096,shared64,shsplit,pbf16,batchfull"),
        ]
        for topo, bs_a, pps_a, d2_a, bd4_a, sw_a, sh_a, mode_a in aot_w18:
            phys_r = 128 if bd4_a else 256
            tpu_devs = topologies.get_topology_desc(topo, platform="tpu").devices
            mesh = Mesh(np.array(tpu_devs[:1]), ("x",))
            s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
            abs_in = (
                s((1, bs_a, 8, 2, 128), jnp.bfloat16),
                s((bs_a, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a, 512 if bd4_a else 1024), jnp.bfloat16),
                s((bs_a * pps_a + 8, phys_r, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a,), jnp.int32),
                s((bs_a * pps_a,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((4 if sh_a else 3,), jnp.int32),
                s((1,), jnp.int32),
            )
            with patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
                comp = jax.jit(
                    jax.shard_map(
                        lambda q_, km_, gt_, c_, kl_, pi_, cq_, ckv_, dt_, sp_, m=mode_a, d2=d2_a, b4=bd4_a, sw=sw_a, us=sh_a: run_decode_fast_rpa(
                            q_, km_, None, c_, kl_, pi_, cq_, ckv_, dt_,
                            sm_scale=64.0 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                            mode_str=m, decode_query_size=1, fused_out_pack=2,
                            prepared_inputs=True, gate=gt_, out_norm_eps=1e-5,
                            kv_pack_dense2=d2, kv_pack_bd4=b4,
                            shared_pages=sp_ if us else None,
                        ),
                        mesh=mesh,
                        in_specs=(P(),) * 10,
                        out_specs=(P(), P()),
                        check_vma=False,
                    )
                ).lower(*abs_in).compile()
            self.assertIsNotNone(comp)
            del comp
            jax.clear_caches()
            gc.collect()

    def test_wave19_swa_g8_g16_and_shared64_c8192_hkv_masked_shsub(self):
        """Verify Wave-19 auto swabatch g8/g16, shared64 c8192+hkv (shared16 non-hkv guard), masked shsub2k tail parity, and TPU7x AOT."""
        import gc
        from unittest.mock import patch

        from jax.experimental import topologies
        from jax.experimental.pallas import tpu as pltpu
        from jax.sharding import Mesh, PartitionSpec as P
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            parse_decode_fast_config,
            run_decode_fast_rpa,
        )

        # 1. Routing & opt-out checks across SWA/1K g8/g16 and shared64 c8192+hkv vs shared16.
        cfg_bd4_1k = parse_decode_fast_config(
            "tp16,w8,c1024,batchfull,dyn4k",
            sliding_window=None, pages_per_seq=4, page_size=512, kv_pack_bd4=True, b_tokens=64,
        )
        self.assertEqual((cfg_bd4_1k.loop_kind, cfg_bd4_1k.num_slots, cfg_bd4_1k.swabatch_group_size), ("swabatch", 3, 16))
        cfg_d2_1k = parse_decode_fast_config(
            "w8,batchfull,noadapt,hoistkv,g4k,foldcol,batch2,batch4",
            sliding_window=None, pages_per_seq=4, page_size=512, kv_pack_dense2=True, b_tokens=64,
        )
        self.assertEqual((cfg_d2_1k.loop_kind, cfg_d2_1k.num_slots, cfg_d2_1k.swabatch_group_size), ("swabatch", 4, 8))
        cfg_bd4_256 = parse_decode_fast_config(
            "tp16,w8,c1024,batchfull,dyn4k",
            sliding_window=2049, pages_per_seq=8, page_size=512, kv_pack_bd4=True, b_tokens=256,
        )
        self.assertEqual((cfg_bd4_256.loop_kind, cfg_bd4_256.num_slots, cfg_bd4_256.swabatch_group_size), ("swabatch", 3, 16))
        cfg_sh16 = parse_decode_fast_config(
            "tp16,w8,c4096,shared64,q256,pbf16,batchfull",
            sliding_window=None, pages_per_seq=529, page_size=512, shared_group=16, kv_pack_bd4=True, b_tokens=16,
        )
        self.assertEqual((cfg_sh16.chunk_sz, cfg_sh16.hoist_new_kv), (8192, False))
        cfg_sh64_opt = parse_decode_fast_config(
            "w8,c4096,shared64,shc4096,nohoistkv,batchfull",
            sliding_window=None, pages_per_seq=529, page_size=512, shared_group=64, kv_pack_dense2=True, b_tokens=64,
        )
        self.assertEqual((cfg_sh64_opt.chunk_sz, cfg_sh64_opt.hoist_new_kv), (4096, False))

        def _poison_kvc(kvc_u8, pg_mat, kvl_np, start_pg, bd4, n_act):
            pps = pg_mat.shape[1]
            for s in range(pg_mat.shape[0]):
                end_tok = max(int(kvl_np[s]) - 1, 0) if s < n_act else 0
                hi_pg = end_tok // 512
                for p in range(start_pg, pps):
                    if p > hi_pg or s >= n_act:
                        kvc_u8[pg_mat[s, p], :, :, :, :] = 0x7F
                    elif p == hi_pg:
                        for t in range(end_tok % 512, 512):
                            if bd4:
                                r, pair, c0 = t // 4, (t % 4) // 2, (t % 2) * 64
                                kvc_u8[pg_mat[s, p], r, 0, pair * 2 : pair * 2 + 2, c0 : c0 + 64] = 0x7F
                            else:
                                kvc_u8[pg_mat[s, p], t // 2, 0, (t % 2) * 2 :, :] = 0x7F

        rng = np.random.default_rng(20261901)
        ip_race = pltpu.InterpretParams(detect_races=True, uninitialized_memory="nan")

        # 2. Eager race + NaN + 0x7F parity for B=64 BD4 SWA2K g16 & B=64 D2 1K_FULL g8.
        for case_tag, bd4, sw, pps_s, base_m in (
            ("bd4_swa_g16", True, 2049, 6, "tp16,w8,c1024,batchfull,dyn4k"),
            ("d2_1k_g8", False, None, 4, "w8,batchfull,c1024,dyn4k"),
        ):
            B_s, n_act_s = 64, 60
            d2 = not bd4
            phys_rows = 128 if bd4 else 256
            cols = 512 if bd4 else 1024
            kvl_np = rng.integers(300, 2000 if sw else 1024, size=(B_s,), dtype=np.int32)
            kvl_np[0], kvl_np[1] = 1, 513
            pg_mat = rng.permutation(B_s * pps_s).astype(np.int32).reshape(B_s, pps_s)
            kvc_u8 = rng.integers(0x30, 0x3D, size=(B_s * pps_s + 4, phys_rows, 1, 4, 128), dtype=np.uint8)
            _poison_kvc(kvc_u8, pg_mat, kvl_np, 0, bd4, n_act_s)
            kvc = jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_5d = jnp.asarray(rng.standard_normal((1, B_s, 8, 2, 128)) * 0.1, jnp.bfloat16)
            kvm = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B_s, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gate = jnp.asarray(rng.standard_normal((B_s, cols)) * 0.1, jnp.bfloat16)
            kvl = jnp.asarray(kvl_np, jnp.int32)
            pg = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
            cuq = jnp.asarray(np.concatenate([np.arange(n_act_s + 1), np.full((B_s - n_act_s,), n_act_s)], dtype=np.int32))
            cukv = jnp.arange(B_s + 1, dtype=jnp.int32) * (pps_s * 512)
            dist = jnp.asarray([n_act_s, n_act_s, n_act_s], jnp.int32)
            ref_o, ref_kv = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                mode_str=f"{base_m},b1", decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, interpret=ip_race,
            )
            grp_o, grp_kv = run_decode_fast_rpa(
                q_5d, kvm, None, kvc, kvl, pg, cuq, cukv, dist,
                sm_scale=64.0 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                mode_str=base_m, decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gate, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, interpret=ip_race,
            )
            r_np, g_np = np.asarray(ref_o, np.float64).ravel(), np.asarray(grp_o, np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(g_np)))
            cos_w = float(np.dot(r_np, g_np) / (np.linalg.norm(r_np) * np.linalg.norm(g_np) + 1e-12))
            self.assertGreater(cos_w, 0.99999, f"{case_tag} cos={cos_w}")
            self.assertTrue(np.array_equal(np.asarray(ref_kv).view(np.uint8), np.asarray(grp_kv).view(np.uint8)))

        # 3. Eager race + NaN + 0x7F masked shsub2k tail (valid=6144 in 8192 blk) + hkv parity for shared64 (D2 & BD4).
        B_sh, sh_pgs, pps_sh = 64, 12, 24  # sh_pgs*512=6144 in 8192 blk + B=64 -> triggers mask_valid=True + shsub2k + hoistkv!
        kvl_sh = np.full((B_sh,), sh_pgs * 512 + 300, dtype=np.int32)
        kvl_sh[0], kvl_sh[1] = sh_pgs * 512, sh_pgs * 512 + 123
        for d2, bd4 in ((True, False), (False, True)):
            phys_rows = 128 if bd4 else 256
            cols = 512 if bd4 else 1024
            nb_s = sh_pgs + B_sh * (pps_sh - sh_pgs) + 8
            pg_mat_s = np.zeros((B_sh, pps_sh), dtype=np.int32)
            pg_mat_s[:, :sh_pgs] = np.arange(sh_pgs, dtype=np.int32)
            pg_mat_s[:, sh_pgs:] = (sh_pgs + rng.permutation(B_sh * (pps_sh - sh_pgs)).astype(np.int32)).reshape(B_sh, pps_sh - sh_pgs)
            kvc_s_u8 = rng.integers(0x30, 0x3D, size=(nb_s, phys_rows, 1, 4, 128), dtype=np.uint8)
            _poison_kvc(kvc_s_u8, pg_mat_s, kvl_sh, sh_pgs, bd4, B_sh)
            kvc_s = jax.lax.bitcast_convert_type(jnp.asarray(kvc_s_u8, jnp.uint8), jnp.float8_e4m3fn)
            q_s5 = jnp.asarray(rng.standard_normal((1, B_sh, 8, 2, 128)) * 0.1, jnp.bfloat16)
            km_s = jax.lax.bitcast_convert_type(
                jnp.asarray(rng.integers(0x30, 0x3D, size=(B_sh, 1, 4, 128), dtype=np.uint8), jnp.uint8),
                jnp.float8_e4m3fn,
            )
            gt_s = jnp.asarray(rng.standard_normal((B_sh, cols)) * 0.1, jnp.bfloat16)
            kvl_s = jnp.asarray(kvl_sh, jnp.int32)
            pg_s = jnp.asarray(pg_mat_s.reshape(-1), jnp.int32)
            cuq_s = jnp.arange(B_sh + 1, dtype=jnp.int32)
            ckv_s = jnp.arange(B_sh + 1, dtype=jnp.int32) * (pps_sh * 512)
            dist_s = jnp.asarray([B_sh, B_sh, B_sh], jnp.int32)
            sp_s = jnp.asarray([sh_pgs], jnp.int32)
            base_sh = ("tp16," if bd4 else "") + "w8,c4096,shared64,pbf16,batchfull"
            ref_so, ref_sk = run_decode_fast_rpa(
                q_s5, km_s, None, kvc_s, kvl_s, pg_s, cuq_s, ckv_s, dist_s,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=f"{base_sh},noshsub,nofold", decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gt_s, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, shared_pages=sp_s, interpret=ip_race,
            )
            sub_so, sub_sk = run_decode_fast_rpa(
                q_s5, km_s, None, kvc_s, kvl_s, pg_s, cuq_s, ckv_s, dist_s,
                sm_scale=64.0 ** -0.5, sliding_window=None, k_scale=1.0, v_scale=1.0,
                mode_str=f"{base_sh},hoistkv", decode_query_size=1, fused_out_pack=2, prepared_inputs=True,
                gate=gt_s, out_norm_eps=1e-5, kv_pack_dense2=d2, kv_pack_bd4=bd4, shared_pages=sp_s, interpret=ip_race,
            )
            rs_np, ss_np = np.asarray(ref_so, np.float64).ravel(), np.asarray(sub_so, np.float64).ravel()
            self.assertTrue(np.all(np.isfinite(ss_np)))
            cos_sh = float(np.dot(rs_np, ss_np) / (np.linalg.norm(rs_np) * np.linalg.norm(ss_np) + 1e-12))
            self.assertGreater(cos_sh, 0.99999, f"masked_shsub bd4={bd4} cos={cos_sh}")
            self.assertTrue(np.array_equal(np.asarray(ref_sk).view(np.uint8), np.asarray(sub_sk).view(np.uint8)))

        # 4. TPU7x AOT (<32 MiB VMEM) for 1K g8/g16 & SWA2K g8/g16.
        aot_w19 = [
            ("tpu7x:2x2x2",  64, 4, False, True,  None, "tp16,w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x1",  64, 4, True,  False, None, "w8,c1024,batchfull,dyn4k"),
            ("tpu7x:2x2x1", 128, 8, True,  False, 2049, "w8,c1024,batchfull,dyn4k"),
        ]
        for topo, bs_a, pps_a, d2_a, bd4_a, sw_a, mode_a in aot_w19:
            phys_r = 128 if bd4_a else 256
            tpu_devs = topologies.get_topology_desc(topo, platform="tpu").devices
            mesh = Mesh(np.array(tpu_devs[:1]), ("x",))
            s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
            abs_in = (
                s((1, bs_a, 8, 2, 128), jnp.bfloat16),
                s((bs_a, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a, 512 if bd4_a else 1024), jnp.bfloat16),
                s((bs_a * pps_a + 8, phys_r, 1, 4, 128), jnp.float8_e4m3fn),
                s((bs_a,), jnp.int32),
                s((bs_a * pps_a,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((bs_a + 1,), jnp.int32),
                s((3,), jnp.int32),
            )
            with patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
                comp = jax.jit(
                    jax.shard_map(
                        lambda q_, km_, gt_, c_, kl_, pi_, cq_, ckv_, dt_, m=mode_a, d2=d2_a, b4=bd4_a, sw=sw_a: run_decode_fast_rpa(
                            q_, km_, None, c_, kl_, pi_, cq_, ckv_, dt_,
                            sm_scale=64.0 ** -0.5, sliding_window=sw, k_scale=1.0, v_scale=1.0,
                            mode_str=m, decode_query_size=1, fused_out_pack=2,
                            prepared_inputs=True, gate=gt_, out_norm_eps=1e-5,
                            kv_pack_dense2=d2, kv_pack_bd4=b4,
                        ),
                        mesh=mesh,
                        in_specs=(P(),) * 9,
                        out_specs=(P(), P()),
                        check_vma=False,
                    )
                ).lower(*abs_in).compile()
            self.assertIsNotNone(comp)
            del comp
            jax.clear_caches()
            gc.collect()


if __name__ == "__main__":
    unittest.main()
