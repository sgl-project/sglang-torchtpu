"""Tests for the Muse megakernel's per-step config selection (host-only; needs jax)."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
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


def _megakernel(**flags):
    return SimpleNamespace(
        geo=_GEOMETRY,
        tp_size=8,
        # Layers 0 and 61 hold FP8 experts quantized at load; the rest are NVFP4.
        kinds=("fp8",) + ("fp4",) * (_LAYERS - 2) + ("fp8",),
        max_position=262144,
        chunk_pages=4,
        kv_packed=flags.get("kv_packed", True),
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


if __name__ == "__main__":
    unittest.main()
