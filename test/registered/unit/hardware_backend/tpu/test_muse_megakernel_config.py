"""Tests for the Muse megakernel's per-step config selection (host-only; needs jax)."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.muse_megakernel import kernel as K
from sglang_tpu.muse_megakernel import ops

register_tpu_ci(est_time=2, suite="stage-a-test-tpu-unit", chips=1)

_LAYERS = 62
# Muse Spark at tp8: per-device heads and expert-TP intermediate slice.
_GEOMETRY = SimpleNamespace(
    hidden=8192,
    head_dim=64,
    q_heads=16,
    kv_heads=2,
    n_layers=_LAYERS,
    n_experts=256,
    top_k=8,
    expert_hidden=4096,
    inter_local=512,
    rms_eps=1e-5,
    post_eps=1e-8,
    scale_query_by=5.473,
    attn_scale=0.125,
    router_mult=0.1768,
    route_scale=1.0,
    router_eps=1e-15,
    window=2048,
    global_layers=tuple(range(1, _LAYERS, 4)),
)


# Muse Spark at tp16 (two hosts): one KV head per device, so the pool cannot pack heads.
_GEOMETRY_TP16 = SimpleNamespace(
    **{**vars(_GEOMETRY), "q_heads": 8, "kv_heads": 1, "inter_local": 256}
)


def _megakernel(geo=_GEOMETRY, tp_size=8, **flags):
    return SimpleNamespace(
        geo=geo,
        tp_size=tp_size,
        # Layers 0 and 61 hold FP8 experts quantized at load; the rest are NVFP4.
        kinds=("fp8",) + ("fp4",) * (_LAYERS - 2) + ("fp8",),
        max_position=262144,
        chunk_pages=4,
        kv_packed=flags.get("kv_packed", True),
        scale_fp8=False,
    )


class TestMuseMegakernelConfig(CustomTestCase):
    def test_served_shapes_fit_the_scratch_budget(self):
        # (rows per request, requests): decode and DFlash verify blocks up to the
        # 32-row cap; 8 rows of 4 is the shape whose Mosaic temporaries set the budget.
        for q, n_req in ((1, 1), (1, 8), (8, 1), (8, 2), (4, 8), (8, 4)):
            for flags in ({}, {"kv_packed": False}):
                with self.subTest(q=q, n_req=n_req, **flags):
                    rows = -(-q * n_req // 8) * 8
                    cfg = ops.MuseMegakernel.config(
                        _megakernel(**flags),
                        rows=rows,
                        q=q,
                        n_req=n_req,
                        n_page_entries=n_req * 2048,
                        page_size=128,
                        capture=(1, 16, 30, 44, 59),
                    )
                    self.assertLessEqual(
                        K.scratch_vmem_bytes(cfg), ops._SCRATCH_VMEM_BUDGET
                    )
                    self.assertEqual(cfg.stream_depth, cfg.ring_slots - 1)

    def test_sal_attention_flags(self):
        # The cc=1 verify step on fp8 SEQ_ALONG_LANE pages.
        def cfg():
            return ops.MuseMegakernel.config(
                _megakernel(),
                rows=8,
                q=8,
                n_req=1,
                n_page_entries=4097,
                page_size=256,
                capture=(1, 16, 30, 44, 59),
                kv_sal=True,
                kv_sal_fp8=True,
            ).validate()

        default = cfg()
        # 1024-token chunks: one expert w13 slot row, the ring a view of those slots
        self.assertEqual((default.chunk_pages, default.kv_ring), (4, 32))
        self.assertTrue(default.attn_unshifted and default.attn_pipe)
        self.assertEqual(default.attn_streams, 8)
        bound = _GEOMETRY.scale_query_by * _GEOMETRY.head_dim * _GEOMETRY.attn_scale
        self.assertLessEqual(1.1 * bound, K.UNSHIFTED_MAX)
        self.assertLessEqual(K.scratch_vmem_bytes(default), ops._SCRATCH_VMEM_BUDGET)
        # SWA layers' short chunks are sub-slots of the full-attention ring: same VMEM, no overrun
        with tpu_envs.SGLANG_TPU_MUSE_MK_SAL_SWA_CHUNK_TOKENS.override(512):
            split = cfg()
        self.assertEqual((split.chunk_pages, split.swa_chunk_pages), (4, 2))
        self.assertEqual(
            2 * split.swa_kv_group * split.swa_chunk_pages,
            split.kv_ring * split.chunk_pages,
        )
        self.assertEqual(split.swa_kv_group % split.attn_streams, 0)

    def test_verify_steps_fold_and_decode_steps_do_not(self):
        for q, n_req, fold in (
            (8, 1, True),
            (8, 2, True),
            (1, 1, False),
            (1, 2, False),
        ):
            with self.subTest(q=q, n_req=n_req):
                cfg = ops.MuseMegakernel.config(
                    _megakernel(),
                    rows=-(-q * n_req // 8) * 8,
                    q=q,
                    n_req=n_req,
                    n_page_entries=n_req * 2048,
                    page_size=256,
                    capture=(1, 16, 30, 44, 59),
                    kv_sal=True,
                    kv_sal_fp8=True,
                ).validate()
                self.assertEqual((cfg.fold_f, cfg.fold_o), (fold, fold))
                self.assertEqual(cfg.ring_slots, 10)

    def test_48_row_shapes_fit_the_scratch_budget(self):
        # SGLANG_TPU_MUSE_MK_MAX_ROWS=48: 12 requests x 4 verify tokens and 6 x 8, on the
        # packed bf16 pool and on fp8 SEQ_ALONG_LANE pages (which keep extra headroom).
        for q, n_req in ((4, 12), (8, 6)):
            for kv_sal in (False, True):
                with self.subTest(q=q, n_req=n_req, kv_sal=kv_sal):
                    cfg = ops.MuseMegakernel.config(
                        _megakernel(),
                        rows=48,
                        q=q,
                        n_req=n_req,
                        n_page_entries=n_req * 4096,
                        page_size=256 if kv_sal else 128,
                        capture=(1, 16, 30, 44, 59),
                        kv_sal=kv_sal,
                        kv_sal_fp8=kv_sal,
                    ).validate()
                    budget = ops._SCRATCH_VMEM_BUDGET - (
                        ops._SAL_VMEM_HEADROOM if kv_sal else 0
                    )
                    self.assertLessEqual(K.scratch_vmem_bytes(cfg), budget)
                    self.assertEqual(cfg.compact_rows, 8)

    def test_raised_row_cap_shapes_fit_the_scratch_budget(self):
        # SGLANG_TPU_MUSE_MK_MAX_ROWS: 12, 16 and 24 requests of a 4-token DFlash verify block on the
        # SEQ_ALONG_LANE fp8 pools the AgentX recipe serves.
        for n_req in (12, 16, 24):
            with self.subTest(n_req=n_req):
                cfg = ops.MuseMegakernel.config(
                    _megakernel(kv_packed=False),
                    rows=4 * n_req,
                    q=4,
                    n_req=n_req,
                    n_page_entries=n_req * 2048,
                    page_size=256,
                    capture=(1, 16, 30, 44, 59),
                    kv_sal=True,
                    kv_sal_fp8=True,
                ).validate()
                self.assertLessEqual(
                    K.scratch_vmem_bytes(cfg), ops._SCRATCH_VMEM_BUDGET
                )

    def test_tp16_shapes_keep_a_deep_weight_ring(self):
        # tp16's dense weights split into 26 tiles, whose only ring divisor is 2; the
        # tiling must split tiles until a ring of at least MIN_RING_SLOTS divides them.
        mk = _megakernel(geo=_GEOMETRY_TP16, tp_size=16, kv_packed=False)
        for q, n_req in ((1, 1), (4, 1), (4, 4), (4, 8), (8, 4)):
            with self.subTest(q=q, n_req=n_req):
                cfg = ops.MuseMegakernel.config(
                    mk,
                    rows=-(-q * n_req // 16) * 16,
                    q=q,
                    n_req=n_req,
                    n_page_entries=n_req * 2048,
                    page_size=128,
                    capture=(1, 16, 30, 44, 59),
                ).validate()
                self.assertGreaterEqual(cfg.ring_slots, K.MIN_RING_SLOTS)
                self.assertLessEqual(
                    K.scratch_vmem_bytes(cfg), ops._SCRATCH_VMEM_BUDGET
                )

    def test_block_send_follows_the_env_and_yields_to_y_early(self):
        def config(rows, n_req):
            return ops.MuseMegakernel.config(
                _megakernel(kv_packed=False),
                rows=rows,
                q=4,
                n_req=n_req,
                n_page_entries=n_req * 2048,
                page_size=256,
                capture=(1, 16, 30, 44, 59),
                kv_sal=True,
                kv_sal_fp8=True,
            )

        with tpu_envs.SGLANG_TPU_MK_Y_BLOCK_SEND.override(False):
            self.assertFalse(config(32, 8).y_block_send)
        with tpu_envs.SGLANG_TPU_MK_Y_BLOCK_SEND.override(True):
            self.assertTrue(config(32, 8).y_block_send)
            self.assertTrue(config(64, 16).y_block_send)
            with tpu_envs.SGLANG_TPU_MK_Y_EARLY.override(True):
                # y_early owns the small steps' sends; the larger steps keep block sends
                self.assertTrue(config(16, 4).y_early)
                self.assertFalse(config(16, 4).y_block_send)
                self.assertTrue(config(32, 8).y_block_send)


if __name__ == "__main__":
    unittest.main()
