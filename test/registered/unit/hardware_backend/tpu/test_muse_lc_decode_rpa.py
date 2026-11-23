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


if __name__ == "__main__":
    unittest.main()
