# SPDX-License-Identifier: Apache-2.0
"""Unit, CPU interpret-mode (0x7F NaN & DMA race), and TPU7x Mosaic export tests for Wave-7 RPA."""

import ast
import os
import unittest
from unittest.mock import patch

import jax
from jax import export as jax_export
import jax.numpy as jnp
import numpy as np

from test.registered.unit.hardware_backend.tpu.test_muse_w2_attn_rpa import (
    _cosine,
    _exact_decode_ref,
    _fp8_from_u8,
)


class TestMuseWave7AttnRpa(unittest.TestCase):
    def test_w7_flags_default_off_and_preserve_stock_behavior(self):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            can_use_decode_fast,
            parse_decode_fast_config,
        )
        from sglang_tpu.envs import envs

        self.assertFalse(envs.SGLANG_TPU_RPA_V3_DECODE_FAST_W7.get())
        self.assertEqual(envs.SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS.get(), "")

        q_j = jnp.zeros((64, 16, 128), dtype=jnp.bfloat16)
        c_j = jnp.zeros((8300, 128, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        # Stock behavior: mode_str="1" rejects pps=129 > 128 when DECODE_FAST_W7 is False
        self.assertFalse(
            can_use_decode_fast(
                mode_str="1",
                q=q_j,
                kv_cache_fused=c_j,
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
        cfg_stock = parse_decode_fast_config("1", sliding_window=2048, pages_per_seq=129)
        self.assertEqual(cfg_stock.loop_kind, "stream0b")
        self.assertFalse(cfg_stock.lazy_slot_zero)
        self.assertFalse(cfg_stock.pv_bf16)

    def test_no_bare_envfield_bool_eval_in_w7_touched_files(self):
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../../")
        )
        rels = (
            "sglang_tpu/envs.py",
            "sglang_tpu/attention_pallas_rpa_v3.py",
            "sglang_tpu/attention/pallas/rpa_v3_decode_fast.py",
            "sglang_tpu/attention/pallas/ragged_paged_attention_v3.py",
        )
        for rel in rels:
            with open(os.path.join(repo_root, rel)) as fh:
                tree = ast.parse(fh.read(), filename=rel)
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    test_node = node.test
                    if (
                        isinstance(test_node, ast.Attribute)
                        and isinstance(test_node.value, ast.Name)
                        and test_node.value.id in ("envs", "tpu_envs")
                        and test_node.attr.startswith("SGLANG_")
                    ):
                        self.fail(
                            f"Bare EnvField truthiness evaluation in {rel}:{node.lineno}"
                        )

    def test_decode_fast_w7_and_prefill_w7_preset_routing(self):
        from sglang_tpu import attention_pallas_rpa_v3 as rpa_v3_mod
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import (
            can_use_decode_fast,
            parse_decode_fast_config,
        )
        from sglang_tpu.envs import envs

        q_dec = jnp.zeros((1, 64, 8, 2, 128), dtype=jnp.bfloat16)
        q_3d = jnp.zeros((64, 16, 128), dtype=jnp.bfloat16)
        c_129 = jnp.zeros((64 * 129 + 8, 128, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        c_256 = jnp.zeros((64 * 256 + 8, 128, 1, 4, 128), dtype=jnp.float8_e4m3fn)
        kl_dec = jnp.full((64,), 8500, dtype=jnp.int32)
        pi_129 = jnp.arange(64 * 129, dtype=jnp.int32)

        with (
            envs.SGLANG_TPU_RPA_V3_DECODE_FAST.override("1"),
            envs.SGLANG_TPU_RPA_V3_DECODE_FAST_V2.override(True),
            envs.SGLANG_TPU_RPA_V3_DECODE_FAST_W7.override(True),
            envs.SGLANG_TPU_RPA_V3_PREFILL_KERNEL_MODE.override("m0_ring_fuse"),
            envs.SGLANG_TPU_RPA_V3_PREFILL_W7_OPTS.override("w7"),
        ):
            # 1. Decode: pps=129 & pps=256 both accepted; _rpa_v3_extra_kwargs folds "1" -> "w7"
            for pps, cj in ((129, c_129), (256, c_256)):
                self.assertTrue(
                    can_use_decode_fast(
                        mode_str="1",
                        q=q_3d,
                        kv_cache_fused=cj,
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
                    )
                )
            cfg_swa = parse_decode_fast_config("w7", sliding_window=2048, pages_per_seq=129)
            self.assertEqual(cfg_swa.loop_kind, "swabatch")
            self.assertFalse(cfg_swa.lazy_slot_zero)
            self.assertFalse(cfg_swa.pv_bf16)

            cfg_129 = parse_decode_fast_config("w7", sliding_window=None, pages_per_seq=129)
            self.assertEqual(cfg_129.loop_kind, "w7auto")
            self.assertEqual(cfg_129.chunk_sz, 5120)
            self.assertEqual(cfg_129.max_ring_pages, 160)
            self.assertFalse(cfg_129.pv_bf16)
            self.assertTrue(cfg_129.lazy_slot_zero)

            cfg_256 = parse_decode_fast_config("w7", sliding_window=None, pages_per_seq=256)
            self.assertEqual(cfg_256.loop_kind, "ring4k")
            self.assertEqual(cfg_256.num_slots, 4)
            self.assertEqual(cfg_256.chunk_sz, 4096)
            self.assertTrue(cfg_256.coalesce_dma)
            self.assertFalse(cfg_256.pv_bf16)

            def _capture_kwargs(
                *, sliding_window, skip_decode, skip_mixed, queries, keys, kv_lens, page_indices
            ):
                cap = {}
                rpa_v3_mod._invoke_v3_kernel(
                    lambda *a, **kw: (cap.update(kw), (a[0], a[3]))[1],
                    queries=queries,
                    keys=keys,
                    values=None,
                    kv_cache_fused=c_129,
                    kv_lens=kv_lens,
                    page_indices=page_indices,
                    cu_q_lens=jnp.arange(kv_lens.shape[0] + 1, dtype=jnp.int32),
                    cu_kv_lens=jnp.arange(kv_lens.shape[0] + 1, dtype=jnp.int32),
                    distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
                    attention_sink=None,
                    sm_scale=0.125,
                    sliding_window=sliding_window,
                    soft_cap=None,
                    k_scale=1.0,
                    v_scale=1.0,
                    decode_query_size=1,
                    prepared_inputs=True,
                    skip_decode=skip_decode,
                    skip_mixed=skip_mixed,
                )
                return cap

            dec_kw = _capture_kwargs(
                sliding_window=None,
                skip_decode=False,
                skip_mixed=True,
                queries=q_dec,
                keys=jnp.zeros((64, 1, 4, 128), dtype=jnp.float8_e4m3fn),
                kv_lens=kl_dec,
                page_indices=pi_129,
            )
            self.assertEqual(dec_kw["_decode_fast"], "w7")
            with envs.SGLANG_TPU_RPA_V3_DECODE_FAST.override("streamdyn"):
                dec_sdyn_kw = _capture_kwargs(
                    sliding_window=None,
                    skip_decode=False,
                    skip_mixed=True,
                    queries=q_dec,
                    keys=jnp.zeros((64, 1, 4, 128), dtype=jnp.float8_e4m3fn),
                    kv_lens=kl_dec,
                    page_indices=pi_129,
                )
                self.assertEqual(dec_sdyn_kw["_decode_fast"], "w7")

            # 2. Prefill "w7" preset across T=8192 and T=16384 (chunk16k):
            # SWA gets (256, 2048) + {fp8_qk, mrb3}; Full gets lossless (512, 4096) tqloop!
            for t_q in (8192, 16384):
                q_p = jnp.zeros((1, t_q, 8, 2, 128), dtype=jnp.bfloat16)
                k_p = jnp.zeros((t_q, 1, 4, 128), dtype=jnp.float8_e4m3fn)
                kl_p = jnp.zeros((8,), dtype=jnp.int32)
                pi_p = jnp.arange(8 * 129, dtype=jnp.int32)
                sw_kw = _capture_kwargs(
                    sliding_window=2048,
                    skip_decode=True,
                    skip_mixed=False,
                    queries=q_p,
                    keys=k_p,
                    kv_lens=kl_p,
                    page_indices=pi_p,
                )
                self.assertEqual(sw_kw["m_block_sizes"], (256, 2048, 256, 256))
                self.assertIn("fp8_qk", sw_kw["_prefill_opts"])
                self.assertIn("mrb3", sw_kw["_prefill_opts"])

                fl_kw = _capture_kwargs(
                    sliding_window=None,
                    skip_decode=True,
                    skip_mixed=False,
                    queries=q_p,
                    keys=k_p,
                    kv_lens=kl_p,
                    page_indices=pi_p,
                )
                self.assertEqual(fl_kw["m_block_sizes"], (512, 4096, 256, 256))
                self.assertNotIn("fp8_qk", fl_kw["_prefill_opts"])
                self.assertNotIn("tq_unroll", fl_kw["_prefill_opts"])

    def test_w7auto_ring4k_and_swafast_cpu_interpret_nan_and_race_free(self):
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa

        rng = np.random.default_rng(707)
        b_sz = 8
        pps = 129
        n_pages = b_sz * pps + 4
        # Poison ALL unvisited KV cache entries with 0x7F (FP8 NaN)!
        cache_u8 = np.full((n_pages, 128, 1, 4, 128), 0x7F, dtype=np.uint8)
        valid_u8 = rng.integers(0, 0x70, size=cache_u8.shape, dtype=np.uint8)
        # Non-monotonic sequence lengths exercising <=1K, ==1, 1K..8K, and >8K on shared slots!
        kv_lens = np.array([300, 8192, 8500, 1, 1400, 410, 9100, 600], dtype=np.int32)
        page_indices = np.arange(b_sz * pps, dtype=np.int32)
        cu_q_lens = np.arange(b_sz + 1, dtype=np.int32)
        cu_kv_lens = np.arange(b_sz + 1, dtype=np.int32) * (pps * 128)
        dist = np.array([b_sz, b_sz, b_sz], dtype=np.int32)

        for s in range(b_sz):
            for pos in range(int(kv_lens[s]) - 1):
                pg = int(page_indices[s * pps + pos // 128])
                cache_u8[pg, pos % 128] = valid_u8[pg, pos % 128]

        q_3d = rng.standard_normal((b_sz, 16, 128)).astype(np.float32) * 0.3
        q_3d[:, :8, 64:] = 0.0
        q_3d[:, 8:, :64] = 0.0
        q_j_3d = jnp.asarray(q_3d, dtype=jnp.bfloat16)
        q_j_prep = q_j_3d.reshape(1, b_sz, 8, 2, 128)
        kv_new_j = _fp8_from_u8(
            rng.integers(0, 0x70, size=(b_sz, 1, 4, 128), dtype=np.uint8)
        )

        for mode_str, sw in (
            ("w7", None),
            ("w7", 2048),
            ("ring4k_s4_pbf16", None),
            ("swafast_pbf16", 2048),
        ):
            with self.subTest(mode_str=mode_str, sw=sw):
                ref_out, ref_cache_u8 = _exact_decode_ref(
                    np.asarray(q_j_3d, dtype=np.float32),
                    np.asarray(kv_new_j),
                    cache_u8,
                    kv_lens,
                    page_indices,
                    cu_q_lens,
                    cu_kv_lens,
                    b_sz,
                    sm_scale=0.125,
                    sliding_window=sw,
                )
                ref_pack2 = np.concatenate(
                    [
                        ref_out[:, :8, :64].reshape(b_sz, 512),
                        ref_out[:, 8:, 64:].reshape(b_sz, 512),
                    ],
                    axis=-1,
                )
                out_2d, upd_cache = run_decode_fast_rpa(
                    q_j_prep,
                    kv_new_j,
                    None,
                    _fp8_from_u8(cache_u8),
                    jnp.asarray(kv_lens),
                    jnp.asarray(page_indices),
                    jnp.asarray(cu_q_lens),
                    jnp.asarray(cu_kv_lens),
                    jnp.asarray(dist),
                    sm_scale=0.125,
                    sliding_window=sw,
                    k_scale=1.0,
                    v_scale=1.0,
                    mode_str=mode_str,
                    fused_out_pack=2,
                    prepared_inputs=True,
                    interpret=True,
                )
                out_np = np.asarray(out_2d, dtype=np.float32)
                self.assertFalse(np.isnan(out_np).any())
                upd_u8 = np.asarray(jax.lax.bitcast_convert_type(upd_cache, jnp.uint8))
                self.assertTrue(np.array_equal(upd_u8, ref_cache_u8))
                self.assertGreater(_cosine(out_np, ref_pack2), 0.9998)

    def test_tpu7x_mosaic_export_all_w7_rpad_and_rpam_kernels(self):
        from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
            ragged_paged_attention,
        )
        from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
        from sglang_tpu.pallas_export import tpu_info_for_cpu_export

        with (
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
            tpu_info_for_cpu_export(),
        ):
            for dtag, dmode, dsw, dpps, expected_tok in (
                ("RPAd_w7_swa", "w7", 2048, 129, "swabatch-mp17"),
                ("RPAd_w7auto_129", "w7", None, 129, "w7auto-mp160"),
                ("RPAd_w7auto_fp8dp_129", "w7auto_fp8dp", None, 129, "fp8dp"),
                ("RPAd_w7_ring4k_256", "w7", None, 256, "ring4k-mp32"),
            ):
                with self.subTest(kernel=dtag):
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q, kv, c, kl, pi, cq, ckv, d, g, _dm=dmode, _sw=dsw: run_decode_fast_rpa(
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
                                sliding_window=_sw,
                                k_scale=1.0,
                                v_scale=1.0,
                                mode_str=_dm,
                                fused_out_pack=2,
                                prepared_inputs=True,
                                gate=g,
                                out_norm_eps=1e-5,
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, 64, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((64, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((16400, 128, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((64,), jnp.int32),
                        jax.ShapeDtypeStruct((64 * dpps,), jnp.int32),
                        jax.ShapeDtypeStruct((65,), jnp.int32),
                        jax.ShapeDtypeStruct((65,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                        jax.ShapeDtypeStruct((64, 1024), jnp.bfloat16),
                    ).mlir_module()
                    self.assertIn("RPAd_fast", hlo)
                    self.assertIn(expected_tok, hlo)

            base_sw = {
                "stationary_q_m0_ring_fuse",
                "skip_empty_cdma",
                "mixed_s3",
                "sw2048_fuse",
            }
            base_fl = {
                "stationary_q_m0_ring_fuse",
                "skip_empty_cdma",
                "mixed_s3",
                "bkv_full_unroll",
            }
            for ptag, t_tok, psw, mb, opts in (
                (
                    "RPAm_sw_fp8_qk_mrb3_16k",
                    16384,
                    2048,
                    (256, 2048, 256, 256),
                    frozenset(base_sw | {"fp8_qk", "mrb3"}),
                ),
                (
                    "RPAm_full_512_4k_tqloop_16k",
                    16384,
                    None,
                    (512, 4096, 256, 256),
                    frozenset(base_fl),
                ),
            ):
                with self.subTest(kernel=ptag):
                    hlo = jax_export.export(
                        jax.jit(
                            lambda q, kv, c, kl, pi, cq, ckv, d, _sw=psw, _mb=mb, _op=opts: ragged_paged_attention(
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
                            )
                        ),
                        platforms=["tpu"],
                    )(
                        jax.ShapeDtypeStruct((1, t_tok, 8, 2, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((t_tok, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((1048, 128, 1, 4, 128), jnp.float8_e4m3fn),
                        jax.ShapeDtypeStruct((8,), jnp.int32),
                        jax.ShapeDtypeStruct((8 * 129,), jnp.int32),
                        jax.ShapeDtypeStruct((9,), jnp.int32),
                        jax.ShapeDtypeStruct((9,), jnp.int32),
                        jax.ShapeDtypeStruct((3,), jnp.int32),
                    ).mlir_module()
                    self.assertIn("RPAm", hlo)


if __name__ == "__main__":
    unittest.main()
